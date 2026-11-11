//! Literal Linux POSIX ACL xattr vectors, independent of the filesystem.
use super::validate_linux_acl;

// Linux ACL xattr version 2: owner rwx, one named user with no access,
// group/mask/other with no access. The named entry keeps this an extended ACL.
const OWNER_ONLY: [u8; 44] = [
    2, 0, 0, 0, // version
    1, 0, 7, 0, 255, 255, 255, 255, // owner
    2, 0, 0, 0, 57, 48, 0, 0, // named user 12345
    4, 0, 0, 0, 255, 255, 255, 255, // group
    16, 0, 0, 0, 255, 255, 255, 255, // mask
    32, 0, 0, 0, 255, 255, 255, 255, // other
];

#[test]
fn owner_only_access_and_default_acl_are_accepted() {
    assert!(validate_linux_acl(&OWNER_ONLY, Some(7)).is_ok());
    assert!(validate_linux_acl(&OWNER_ONLY, None).is_ok());
    let mut named_group = OWNER_ONLY;
    named_group[12] = 8;
    assert!(validate_linux_acl(&named_group, Some(7)).is_ok());
}

#[test]
fn any_group_named_or_other_permission_is_rejected() {
    for position in [14, 22, 30, 38] {
        let mut acl = OWNER_ONLY;
        acl[position] = 4;
        assert!(
            validate_linux_acl(&acl, Some(7)).is_err(),
            "byte {position}"
        );
        assert!(
            validate_linux_acl(&acl, None).is_err(),
            "default byte {position}"
        );
    }
}

#[test]
fn malformed_incomplete_or_inconsistent_acl_is_rejected() {
    let mut bad_version = OWNER_ONLY;
    bad_version[0] = 3;
    let mut bad_permission = OWNER_ONLY;
    bad_permission[6] = 8;
    let mut unknown_tag = OWNER_ONLY;
    unknown_tag[12] = 64;
    let mut invalid_named_id = OWNER_ONLY;
    invalid_named_id[16..20].fill(255);
    let mut duplicate = OWNER_ONLY.to_vec();
    duplicate.extend_from_slice(&OWNER_ONLY[4..12]);
    let mut missing_mask = OWNER_ONLY.to_vec();
    missing_mask.drain(28..36);
    for acl in [
        &OWNER_ONLY[..43],
        &bad_version[..],
        &bad_permission[..],
        &unknown_tag[..],
        &invalid_named_id[..],
        &duplicate[..],
        &missing_mask[..],
    ] {
        assert!(validate_linux_acl(acl, Some(7)).is_err());
    }
    assert!(validate_linux_acl(&OWNER_ONLY, Some(6)).is_err());
}
