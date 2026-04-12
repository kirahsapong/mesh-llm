#[derive(Debug, Clone)]
pub struct OwnerKeypair(mesh_client::OwnerKeypair);

impl OwnerKeypair {
    pub fn generate() -> Self {
        Self(mesh_client::OwnerKeypair::generate())
    }

    pub fn owner_id(&self) -> String {
        self.0.owner_id()
    }

    pub fn from_bytes(signing_bytes: &[u8], encryption_bytes: &[u8]) -> Result<Self, String> {
        mesh_client::OwnerKeypair::from_bytes(signing_bytes, encryption_bytes)
            .map(Self)
            .map_err(|err| err.to_string())
    }

    pub fn signing_bytes(&self) -> &[u8; 32] {
        self.0.signing_bytes()
    }

    pub fn encryption_bytes(&self) -> [u8; 32] {
        self.0.encryption_bytes()
    }

    pub(crate) fn into_inner(self) -> mesh_client::OwnerKeypair {
        self.0
    }
}
