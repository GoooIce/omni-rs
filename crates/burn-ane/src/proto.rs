//! Small protobuf writer for Apple's public Model/NeuralNetwork schemas.
//! https://github.com/apple/coremltools/tree/9.0/mlmodel/format
//! Only emitted fields are implemented; Core ML validates the complete graph.
#[derive(Default)]
pub(crate) struct Message(pub Vec<u8>);
impl Message {
    fn varint(&mut self, mut value: u64) {
        while value >= 128 {
            self.0.push((value as u8) | 128);
            value >>= 7;
        }
        self.0.push(value as u8);
    }
    pub fn uint(&mut self, field: u32, value: u64) {
        self.varint((field as u64) << 3);
        self.varint(value);
    }
    pub fn bytes(&mut self, field: u32, value: &[u8]) {
        self.varint(((field as u64) << 3) | 2);
        self.varint(value.len() as u64);
        self.0.extend_from_slice(value);
    }
    pub fn string(&mut self, field: u32, value: &str) {
        self.bytes(field, value.as_bytes());
    }
    pub fn message(&mut self, field: u32, value: Self) {
        self.bytes(field, &value.0);
    }
    pub fn float(&mut self, field: u32, value: f32) {
        self.varint(((field as u64) << 3) | 5);
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    pub fn shape(&mut self, field: u32, shape: &[usize]) {
        for dim in shape {
            self.uint(field, *dim as u64);
        }
    }
}

pub(crate) fn weights(values: &[f32]) -> Message {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
        .collect();
    let mut message = Message::default();
    message.bytes(2, &bytes); // WeightParams.float16Value
    message
}

pub(crate) fn weights_f32(values: &[f32]) -> Message {
    let mut message = Message::default();
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    message.bytes(1, &bytes);
    message
}

pub(crate) fn feature(name: &str, shape: &[usize]) -> Message {
    let mut array = Message::default();
    array.shape(1, shape);
    array.uint(2, 65568); // ArrayFeatureType.FLOAT32
    let mut kind = Message::default();
    kind.message(5, array);
    let mut feature = Message::default();
    feature.string(1, name);
    feature.message(3, kind);
    feature
}
