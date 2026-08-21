//! The protobuf wire format, written against the specification rather than against examples.
//!
//! # Why this exists as its own module
//!
//! `ommx.rs` carried a hand-rolled decoder that grew a field at a time as instances were met, and it
//! produced **six** separate defects — more than the rest of this crate combined. Two were crashes
//! reachable from eleven and twenty-three bytes; one read variable 0 as "not declared" because proto3
//! omits fields at their default; one shipped documented backwards on five surfaces.
//!
//! That is not six unlucky mistakes. It is one structural fault with six symptoms, and it was in the
//! decoder's **type**:
//!
//! ```ignore
//! fn next(&mut self) -> Option<(u32, Body)>   // the old one
//! ```
//!
//! `None` meant *both* "the message ended" and "the message is corrupt". So a truncated instance
//! parsed as a shorter valid instance, silently, and every corruption case had to be caught by hand
//! somewhere further up — which is exactly the game of whack-a-mole the six defects record. A
//! decoder that cannot say "this input is malformed" forces every caller to guess.
//!
//! ```ignore
//! fn next(&mut self) -> Result<Option<Field<'a>>, WireError>   // this one
//! ```
//!
//! `Ok(None)` is the end. `Err` is malformed, with the byte offset. They are different values, and
//! nothing downstream has to infer which happened.
//!
//! # What the old decoder got wrong beyond that, and this does not
//!
//! - **`fixed32` returned zero.** The old arm was `self.i += 4; Some((field, Body::Varint(0)))` — it
//!   advanced four bytes without a bounds check and **substituted 0 for the real value**. Any 32-bit
//!   field would have read as zero with nothing said. This returns [`Value::Fixed32`] with the value.
//! - **Groups (wire types 3 and 4) silently truncated the message.** They are deprecated, not
//!   impossible, and a reader that stops at one reports a partial message as a complete one. Refused
//!   by name here.
//! - **Varints were not canonical.** Protobuf allows at most ten bytes, and the tenth may only carry
//!   one bit. A longer one is malformed input, not a big number.
//! - **Field number 0 is illegal** and was accepted.
//! - **Packed fields truncated on error.** `while let Some(..)` stops at the first bad byte and
//!   returns the values read so far — so a corrupt packed array became a shorter valid one, the same
//!   fault as the message-level one.
//!
//! # Scope
//!
//! Enough of the wire format to read and write the OMMX subset this crate exchanges, and no more.
//! Every wire type is *handled* — the ones not used are refused explicitly rather than ignored,
//! because "I do not implement this" and "I did not notice this" must not look the same to a caller.

/// Why a byte string is not a valid protobuf message.
///
/// Every variant carries the offset, because "malformed" without a position is a bug report nobody
/// can act on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireError {
    /// The message ended in the middle of a field.
    Truncated { at: usize, needed: usize },
    /// A varint ran past ten bytes, or its tenth byte carried more than one bit.
    ///
    /// Both are malformed rather than large: the wire format cannot express a value that would need
    /// them, so accepting one means accepting a number the writer never wrote.
    VarintNotCanonical { at: usize },
    /// A length prefix that cannot be a length on this machine.
    ///
    /// Separate from `Truncated` on purpose. This is the shape that used to WRAP `usize` and sail
    /// past the bounds check into a slice panic, and naming it keeps that distinct from an honestly
    /// short message.
    LengthOutOfRange { at: usize, len: u64 },
    /// Wire types 3 and 4: the deprecated group encoding.
    GroupsUnsupported { at: usize, field: u32 },
    /// Wire types 6 and 7, which do not exist.
    UnknownWireType { at: usize, wire: u32 },
    /// Field number 0, which the format reserves and no writer may emit.
    FieldNumberZero { at: usize },
}

impl core::fmt::Display for WireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WireError::Truncated { at, needed } => {
                write!(f, "truncated at byte {at}: {needed} more byte(s) were needed")
            }
            WireError::VarintNotCanonical { at } => write!(
                f,
                "the varint at byte {at} is longer than ten bytes or overflows 64 bits, which the \
                 wire format cannot express -- this is malformed input, not a large number"
            ),
            WireError::LengthOutOfRange { at, len } => write!(
                f,
                "the length prefix at byte {at} is {len}, which is not addressable here"
            ),
            WireError::GroupsUnsupported { at, field } => write!(
                f,
                "field {field} at byte {at} uses the deprecated group encoding (wire type 3/4), \
                 which this reader does not implement -- refused rather than skipped, because \
                 skipping it would report a partial message as a whole one"
            ),
            WireError::UnknownWireType { at, wire } => {
                write!(f, "wire type {wire} at byte {at} is not one the format defines")
            }
            WireError::FieldNumberZero { at } => {
                write!(f, "field number 0 at byte {at} is reserved and no writer may emit it")
            }
        }
    }
}

/// One field's payload, by wire type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<'a> {
    /// Wire type 0.
    Varint(u64),
    /// Wire type 1. `double` and `fixed64` arrive here.
    Fixed64(u64),
    /// Wire type 5. `float` and `fixed32` arrive here.
    ///
    /// The old decoder returned a varint of zero for this, discarding the value.
    Fixed32(u32),
    /// Wire type 2: strings, bytes, embedded messages and packed repeated fields.
    Bytes(&'a [u8]),
}

impl Value<'_> {
    /// The `double` this holds, or `None` if it is not a `fixed64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Fixed64(v) => Some(f64::from_bits(*v)),
            _ => None,
        }
    }
    /// The `float` this holds, widened, or `None` if it is not a `fixed32`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::Fixed32(v) => Some(f32::from_bits(*v)),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Varint(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

/// A field number and its payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field<'a> {
    pub number: u32,
    pub value: Value<'a>,
}

/// Walks the fields of a protobuf message.
///
/// Unknown *field numbers* are the caller's to skip — that is forward compatibility, and OMMX has
/// grown fields between 2.0 and 2.6. Unknown *wire types* are refused here, because they mean the
/// bytes are not a message this reader understands, which is a different thing entirely.
pub struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Reader { b, i: 0 }
    }

    /// How many bytes have been consumed. Useful for reporting where a higher layer gave up.
    pub fn position(&self) -> usize {
        self.i
    }

    /// The next field, `Ok(None)` at a clean end, `Err` when the bytes are not a message.
    ///
    /// The three outcomes are three values. That is the whole design.
    ///
    /// Named `read_field` rather than `next`, and deliberately NOT an `Iterator`: `Iterator::next`
    /// returns `Option`, which is precisely the conflation of "finished" with "malformed" that the
    /// old decoder had and that this module exists to remove. Clippy flags the name collision, and
    /// it is right to -- a reader whose failure mode is a truncated-looking success should not wear
    /// the clothes of one that cannot fail.
    pub fn read_field(&mut self) -> Result<Option<Field<'a>>, WireError> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let at = self.i;
        let (key, used) = read_varint(self.b, self.i)?;
        self.i += used;

        let number = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        if number == 0 {
            return Err(WireError::FieldNumberZero { at });
        }

        let value = match wire {
            0 => {
                let (v, u) = read_varint(self.b, self.i)?;
                self.i += u;
                Value::Varint(v)
            }
            1 => Value::Fixed64(u64::from_le_bytes(self.take_array::<8>()?)),
            2 => {
                let lat = self.i;
                let (len, u) = read_varint(self.b, self.i)?;
                self.i += u;
                // `usize::try_from`, never `as usize`. The `as` cast is what wrapped, and a wrapped
                // length passes any bounds check you write after it.
                let len = usize::try_from(len)
                    .map_err(|_| WireError::LengthOutOfRange { at: lat, len })?;
                let end = self
                    .i
                    .checked_add(len)
                    .ok_or(WireError::LengthOutOfRange { at: lat, len: len as u64 })?;
                if end > self.b.len() {
                    return Err(WireError::Truncated { at: lat, needed: end - self.b.len() });
                }
                let s = &self.b[self.i..end];
                self.i = end;
                Value::Bytes(s)
            }
            3 | 4 => return Err(WireError::GroupsUnsupported { at, field: number }),
            5 => Value::Fixed32(u32::from_le_bytes(self.take_array::<4>()?)),
            other => return Err(WireError::UnknownWireType { at, wire: other }),
        };
        Ok(Some(Field { number, value }))
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let end = self
            .i
            .checked_add(N)
            .ok_or(WireError::Truncated { at: self.i, needed: N })?;
        if end > self.b.len() {
            return Err(WireError::Truncated { at: self.i, needed: end - self.b.len() });
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.b[self.i..end]);
        self.i = end;
        Ok(out)
    }
}

/// Read one varint, enforcing the format's own length rule.
///
/// At most ten bytes, and the tenth may carry only one bit — 64 bits do not fit in nine groups of
/// seven, and ten groups hold seventy. A longer encoding, or a tenth byte above 1, is a value the
/// wire format cannot represent, so it is malformed input rather than a number to accept.
fn read_varint(b: &[u8], start: usize) -> Result<(u64, usize), WireError> {
    let mut v: u64 = 0;
    let mut i = start;
    for byte_index in 0..10usize {
        let byte = *b.get(i).ok_or(WireError::Truncated { at: start, needed: 1 })?;
        i += 1;
        if byte_index == 9 {
            // The tenth byte contributes bit 63 only.
            if byte > 0x01 {
                return Err(WireError::VarintNotCanonical { at: start });
            }
            v |= (byte as u64) << 63;
            return Ok((v, i - start));
        }
        v |= ((byte & 0x7f) as u64) << (7 * byte_index);
        if byte & 0x80 == 0 {
            return Ok((v, i - start));
        }
    }
    Err(WireError::VarintNotCanonical { at: start })
}

/// Decode a packed repeated varint field, refusing rather than truncating.
///
/// The old version was `while let Some(..)`, which stopped at the first bad byte and returned what it
/// had — so a corrupt packed array silently became a shorter valid one. Same fault as the
/// message-level `Option`, one level down.
pub fn packed_varints(p: &[u8]) -> Result<Vec<u64>, WireError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < p.len() {
        let (v, u) = read_varint(p, i)?;
        out.push(v);
        i += u;
    }
    Ok(out)
}

/// Decode a packed repeated `double` field.
///
/// A packed `double` array is a whole number of eight-byte groups by construction; a trailing
/// remainder means the bytes are not what they claim to be, and a chunking iterator would have
/// dropped it without a word.
pub fn packed_doubles(p: &[u8]) -> Result<Vec<f64>, WireError> {
    if !p.len().is_multiple_of(8) {
        return Err(WireError::Truncated { at: p.len() - (p.len() % 8), needed: 8 - (p.len() % 8) });
    }
    // `as_chunks::<8>` yields `[u8; 8]` directly, so the length is carried by the TYPE and the
    // `try_into().expect("chunks_exact(8)")` this used to need -- an unreachable panic that still
    // had to be read and dismissed by anyone auditing the parser -- is gone rather than justified.
    Ok(p.as_chunks::<8>().0.iter().map(|&c| f64::from_bits(u64::from_le_bytes(c))).collect())
}

// ---- writing -------------------------------------------------------------------------------
//
// The writer is small and has never been the problem, but it lives beside the reader so the two
// stay in step: the round-trip test below is only meaningful if they are the same module's idea of
// the format.

pub fn put_key(buf: &mut Vec<u8>, field: u32, wire: u32) {
    put_varint(buf, ((field as u64) << 3) | wire as u64);
}

pub fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

pub fn put_varint_field(buf: &mut Vec<u8>, field: u32, v: u64) {
    put_key(buf, field, 0);
    put_varint(buf, v);
}

pub fn put_double_field(buf: &mut Vec<u8>, field: u32, v: f64) {
    put_key(buf, field, 1);
    buf.extend_from_slice(&v.to_bits().to_le_bytes());
}

pub fn put_len_field(buf: &mut Vec<u8>, field: u32, body: &[u8]) {
    put_key(buf, field, 2);
    put_varint(buf, body.len() as u64);
    buf.extend_from_slice(body);
}

pub fn put_str_field(buf: &mut Vec<u8>, field: u32, s: &str) {
    put_len_field(buf, field, s.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect every field, or the first error.
    fn all(b: &[u8]) -> Result<Vec<Field<'_>>, WireError> {
        let mut r = Reader::new(b);
        let mut out = Vec::new();
        while let Some(f) = r.read_field()? {
            out.push(f);
        }
        Ok(out)
    }

    #[test]
    fn an_end_and_a_corruption_are_different_answers() {
        // THE fault the old decoder had, stated as a test. Its `next()` returned `Option`, so a
        // truncated message and a finished one were the same value and a partial instance parsed as
        // a complete shorter one.
        let good = {
            let mut b = Vec::new();
            put_varint_field(&mut b, 1, 150);
            b
        };
        assert_eq!(all(&good).unwrap().len(), 1, "a whole message reads");

        // Same bytes, one short: the varint payload is missing.
        let cut = &good[..good.len() - 1];
        match all(cut) {
            Err(WireError::Truncated { .. }) => {}
            other => panic!("a truncated message must be an error, not a shorter one: {other:?}"),
        }
    }

    #[test]
    fn fixed32_carries_its_value_instead_of_zero() {
        // The old arm was `self.i += 4; Some((field, Body::Varint(0)))` -- four bytes advanced with
        // no bounds check, and the value replaced by zero. A float field would have read as 0.0
        // with nothing said.
        let mut b = Vec::new();
        put_key(&mut b, 7, 5);
        b.extend_from_slice(&1.5f32.to_bits().to_le_bytes());
        let fields = all(&b).unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].number, 7);
        assert_eq!(fields[0].value.as_f32(), Some(1.5), "the value, not zero");

        // And it is bounds-checked: three bytes where four are needed.
        let short = &b[..b.len() - 1];
        assert!(matches!(all(short), Err(WireError::Truncated { .. })));
    }

    #[test]
    fn a_hostile_length_prefix_is_named_rather_than_wrapping() {
        // Eleven bytes that aborted the host process through the C ABI: tag 0x0A, then a length of
        // ~2^64. `self.i + len as usize` wrapped, and the wrapped end passed the bounds check.
        let bytes = [0x0A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        match all(&bytes) {
            Err(WireError::LengthOutOfRange { .. }) | Err(WireError::Truncated { .. }) => {}
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_varint_longer_than_the_format_allows_is_malformed_not_large() {
        // Eleven continuation bytes: the format tops out at ten.
        let mut b = vec![0x08];
        b.extend(std::iter::repeat_n(0xFFu8, 11));
        assert!(matches!(all(&b), Err(WireError::VarintNotCanonical { .. })));

        // Ten bytes whose last carries more than bit 63 is also malformed.
        let mut c = vec![0x08];
        c.extend(std::iter::repeat_n(0xFFu8, 9));
        c.push(0x02);
        assert!(matches!(all(&c), Err(WireError::VarintNotCanonical { .. })));

        // But the largest legal u64 reads exactly.
        let mut d = Vec::new();
        put_varint_field(&mut d, 1, u64::MAX);
        assert_eq!(all(&d).unwrap()[0].value.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn groups_and_impossible_wire_types_are_refused_by_name() {
        // Deprecated, not impossible. The old reader returned None, which reported a partial
        // message as a complete one.
        let mut b = Vec::new();
        put_key(&mut b, 4, 3);
        assert!(matches!(all(&b), Err(WireError::GroupsUnsupported { field: 4, .. })));

        let mut c = Vec::new();
        put_key(&mut c, 4, 6);
        assert!(matches!(all(&c), Err(WireError::UnknownWireType { wire: 6, .. })));
    }

    #[test]
    fn field_number_zero_is_refused() {
        let b = [0x00u8, 0x01];
        assert!(matches!(all(&b), Err(WireError::FieldNumberZero { .. })));
    }

    #[test]
    fn a_packed_array_refuses_rather_than_returning_the_prefix_it_managed() {
        // Three varints, then a trailing continuation byte with nothing after it.
        let mut p = Vec::new();
        for v in [1u64, 300, 70000] {
            put_varint(&mut p, v);
        }
        assert_eq!(packed_varints(&p).unwrap(), vec![1, 300, 70000]);

        p.push(0x80); // a varint that never ends
        match packed_varints(&p) {
            Err(WireError::Truncated { .. }) => {}
            other => panic!("a corrupt packed array must not become a shorter valid one: {other:?}"),
        }

        // Doubles: a partial group is a refusal, where `chunks_exact` would have dropped it.
        let mut d = Vec::new();
        d.extend_from_slice(&1.0f64.to_bits().to_le_bytes());
        assert_eq!(packed_doubles(&d).unwrap(), vec![1.0]);
        d.push(0x00);
        assert!(matches!(packed_doubles(&d), Err(WireError::Truncated { .. })));
    }

    #[test]
    fn every_wire_type_round_trips_through_this_module() {
        // The writer and reader must agree, or the round-trip tests above prove only self-consistency
        // of one half.
        let mut b = Vec::new();
        put_varint_field(&mut b, 1, 0);
        put_varint_field(&mut b, 2, u64::MAX);
        put_double_field(&mut b, 3, -1.5);
        put_str_field(&mut b, 4, "hello");
        put_len_field(&mut b, 5, &[]);
        put_key(&mut b, 6, 5);
        b.extend_from_slice(&2.25f32.to_bits().to_le_bytes());

        let f = all(&b).unwrap();
        assert_eq!(f.len(), 6);
        assert_eq!(f[0].value.as_u64(), Some(0));
        assert_eq!(f[1].value.as_u64(), Some(u64::MAX));
        assert_eq!(f[2].value.as_f64(), Some(-1.5));
        assert_eq!(f[3].value.as_bytes(), Some(&b"hello"[..]));
        assert_eq!(f[4].value.as_bytes(), Some(&[][..]));
        assert_eq!(f[5].value.as_f32(), Some(2.25));
    }

    #[test]
    fn an_unknown_field_number_is_skipped_but_an_unknown_wire_type_is_not() {
        // Forward compatibility is about field NUMBERS -- OMMX grew fields between 2.0 and 2.6 --
        // and it is a different question from whether the bytes are a message at all.
        let mut b = Vec::new();
        put_varint_field(&mut b, 1, 7);
        put_str_field(&mut b, 9999, "a field from a later version");
        put_varint_field(&mut b, 2, 8);
        let f = all(&b).unwrap();
        assert_eq!(f.len(), 3, "the reader hands all three up; the caller ignores 9999");
        assert_eq!(f[2].value.as_u64(), Some(8), "and keeps reading past it");
    }
}
