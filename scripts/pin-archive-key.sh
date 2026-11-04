#!/usr/bin/env bash
# Pins the x0ne archive key in this tree, after the signing ceremony (x0ne-OS
# docs/archive-key.md). Copies the PUBLIC keyring to
# x0ne-infer/archive-key/x0ne-archive-keyring.gpg, sets archive_key::RELEASE
# (include_bytes!), replaces the test that wanted no pin, and rewrites every text
# that says this tree pins no key: docs/cloud.md, docs/attested-channel.md,
# x0ne-infer/THREAT_MODEL.md, README.md, the LIMITS of
# x0ne-engine-cloud/src/up.rs and its golden. Nothing is written unless every check passes and every text is found.
#
# Usage: scripts/pin-archive-key.sh <public/x0ne-archive-keyring.gpg> <primary fingerprint>
#   PIN_DATE=YYYY-MM-DD (default: today, UTC) dates the pin in the texts.
# Refuses: an armored or secret keyring, a primary that is not v4 or not the
# fingerprint given, a user ID containing TEST, a tree that already pins a key,
# and a user ID containing REHEARSAL (any case) unless ALLOW_REHEARSAL_KEY=1:
# a rehearsal pins its throwaway key in a throwaway clone only.
# Then: make ci, read the diff, commit. Needs bash and perl (Digest::SHA).
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <public keyring .gpg> <primary fingerprint>" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec perl -w - "$ROOT" "$1" "$2" "${PIN_DATE:-$(date -u +%F)}" <<'PERL'
use strict;
use Digest::SHA qw(sha1_hex);
my ($root, $keyring, $fpr_in, $date) = @ARGV;
sub refuse { print STDERR "pin-archive-key: $_[0]; nothing written\n"; exit 1 }
$date =~ /^\d{4}-\d{2}-\d{2}$/ or refuse("PIN_DATE $date: expected YYYY-MM-DD");
(my $fpr = uc $fpr_in) =~ s/\s+//g;
$fpr =~ /^[0-9A-F]{40}$/ or refuse("fingerprint \"$fpr_in\": expected 40 hex digits (v4)");

open my $kh, '<:raw', $keyring or refuse("$keyring: $!");
my $k = do { local $/; <$kh> };
close $kh;
defined $k && length $k or refuse("$keyring is empty");
$k =~ /^\s*-----BEGIN/ and refuse("$keyring is ASCII-armored: give the binary public keyring");
length $k <= 65536 or refuse("$keyring: over 64 KiB");

# OpenPGP packets, old and new formats (RFC 9580 section 4.2).
my (@p, $i);
$i = 0;
while ($i < length $k) {
    my $ctb = ord substr $k, $i++, 1;
    $ctb & 0x80 or refuse("$keyring: no OpenPGP packet at byte " . ($i - 1));
    my ($tag, $len);
    my $byte = sub { $i < length $k or refuse("$keyring: truncated"); ord substr $k, $i++, 1 };
    if ($ctb & 0x40) {
        $tag = $ctb & 0x3f;
        my $o = $byte->();
        if    ($o < 192) { $len = $o }
        elsif ($o < 224) { $len = (($o - 192) << 8) + $byte->() + 192 }
        elsif ($o == 255) { $len = unpack 'N', substr($k, $i, 4); $i += 4 }
        else { refuse("$keyring: partial body length") }
    } else {
        $tag = ($ctb >> 2) & 0x0f;
        my $lt = $ctb & 3;
        if    ($lt == 0) { $len = $byte->() }
        elsif ($lt == 1) { $len = unpack 'n', substr($k, $i, 2); $i += 2 }
        elsif ($lt == 2) { $len = unpack 'N', substr($k, $i, 4); $i += 4 }
        else { refuse("$keyring: indeterminate packet length") }
    }
    defined $len && $i + $len <= length $k or refuse("$keyring: truncated packet");
    push @p, [$tag, substr($k, $i, $len)];
    $i += $len;
}
grep({ $_->[0] == 5 || $_->[0] == 7 } @p) and refuse("$keyring holds secret key material");
@p && $p[0][0] == 6 or refuse("$keyring does not start with a public key");
grep({ $_->[0] == 6 } @p) == 1 or refuse("$keyring holds more than one primary key");
my $pk = $p[0][1];
ord(substr $pk, 0, 1) == 4 or refuse("$keyring: the primary is not a v4 key");
my $got = uc sha1_hex("\x99" . pack('n', length $pk) . $pk);
$got eq $fpr or refuse("$keyring: its primary is $got, not $fpr");
my @uids = map { $_->[1] } grep { $_->[0] == 13 } @p;
@uids or refuse("$keyring has no user ID");
/TEST/ and refuse("$keyring: user ID \"$_\" says TEST") for @uids;
my $rehearsal = grep { /rehearsal/i } @uids;
$rehearsal and ($ENV{ALLOW_REHEARSAL_KEY} // '') ne '1'
    and refuse("$keyring: user ID \"" . (grep { /rehearsal/i } @uids)[0] . "\" says REHEARSAL: never the release pin; "
        . "ALLOW_REHEARSAL_KEY=1 pins it, in a throwaway clone");

# Edits, in memory; each anchor must be found exactly once.
my %new;
sub file {
    my $rel = shift;
    return $new{$rel} if exists $new{$rel};
    open my $h, '<:raw', "$root/$rel" or refuse("$rel: $!");
    local $/;
    return $new{$rel} = <$h>;
}
sub swap {
    my ($rel, $old, $rep) = @_;
    my $s = file($rel);
    my $n = () = $s =~ /\Q$old\E/g;
    $n == 1 or refuse("$rel: the text to replace is there $n times, not once: \"" . substr($old, 0, 60) . "...\"");
    my $at = index $s, $old;
    substr($s, $at, length $old) = $rep;
    $new{$rel} = $s;
}
# A here-document ends in a newline; an anchor that ends mid-line drops it.
sub midline { my $s = shift; chomp $s; $s }
my $ring = 'x0ne-infer/archive-key/x0ne-archive-keyring.gpg';
-e "$root/$ring" and refuse("$ring exists: this tree already pins a key");
file('x0ne-infer/src/archive_key.rs') =~ /^pub const RELEASE: Option<Pin> = None;$/m
    or refuse('x0ne-infer/src/archive_key.rs: RELEASE is not None, this tree already pins a key');

swap('x0ne-infer/src/archive_key.rs', <<'OLD', <<"NEW");
/// The release pin: None in a tree made before the signing ceremony.
pub const RELEASE: Option<Pin> = None;
OLD
/// The release pin of the signing ceremony, written on $date by
/// scripts/pin-archive-key.sh. A tree made before it had None.
pub const RELEASE: Option<Pin> = Some(Pin {
    primary_fingerprint: "$fpr",
    keyring: include_bytes!("../archive-key/x0ne-archive-keyring.gpg"),
});
NEW
swap('x0ne-infer/src/archive_key.rs', <<'OLD', <<"NEW");
    #[test]
    fn no_release_key_refuses_everything() {
        assert!(
            RELEASE.is_none(),
            "a pin landed: replace this test by one on its fingerprint"
        );
OLD
    /// The pin of $date passes `pinned`; a build without one still
    /// refuses with NO_KEY.
    #[test]
    fn the_release_pin_is_the_ceremony_key() {
        let Some(pin) = &RELEASE else {
            panic!("RELEASE is None")
        };
        assert_eq!(
            pin.primary_fingerprint,
            "$fpr"
        );
        let cert = pinned(pin).expect("the keyring is the key written down");
        assert_eq!(
            cert.primary_fingerprint(),
            "$fpr"
        );
NEW
# A single-quoted here-document keeps backslashes as they are; a double-quoted one halves them.
swap('x0ne-engine-cloud/src/up.rs', <<'OLD', <<"NEW");
     pinned in its build, and a relay built before the signing ceremony pins none and \
     refuses them all; whoever holds the signing subkey can sign any image, and a \
     revocation reaches you only with a new release;",
OLD
     pinned in its build, primary $fpr; \\
     whoever holds the signing subkey can sign any image, and a revocation reaches you \\
     only with a new release;",
NEW
swap('x0ne-engine-cloud/tests/golden/colors_forced.ansi',
    'pinned in its build, and a relay built before the signing ceremony pins none and refuses them all;',
    "pinned in its build, primary $fpr;");
swap('docs/cloud.md', <<'OLD', <<"NEW");
signing ceremony". **This source tree pins no archive key**: `RELEASE` is
`None`. A key revoked after a release keeps verifying in that release until
OLD
signing ceremony". **This source tree pins the archive key** with primary
`$fpr`, since $date
(keyring `x0ne-infer/archive-key/x0ne-archive-keyring.gpg`). A key revoked
after a release keeps verifying in that release until
NEW
swap('docs/cloud.md', <<'OLD', <<"NEW");
  archive key pinned in its build, and a relay built before the signing
  ceremony pins none and refuses them all; whoever holds the signing subkey
  can sign any image, and a revocation reaches you only with a new release;
OLD
  archive key pinned in its build, primary
  `$fpr`; whoever holds the signing subkey
  can sign any image, and a revocation reaches you only with a new release;
NEW
swap('x0ne-infer/THREAT_MODEL.md',
    'This source tree pins no archive key (`archive_key::RELEASE` is `None`).',
    "This source tree pins the archive key with primary `$fpr` since $date (`archive_key::RELEASE`, keyring `x0ne-infer/archive-key/x0ne-archive-keyring.gpg`).");
# Released entries are history: the pin is a new Unreleased entry.
swap('CHANGELOG.md', "## Unreleased\n\n", <<"NEW");
## Unreleased

- The x0ne archive key is pinned since $date, primary
  `$fpr` (`scripts/pin-archive-key.sh`,
  keyring `x0ne-infer/archive-key/x0ne-archive-keyring.gpg`). A release
  build made before that pins none and refuses every reference with 99.
NEW
swap('docs/attested-channel.md', <<'OLD', <<"NEW");
`reference::load`; 99, and a build made before the signing ceremony pins no
key and refuses every reference, see
OLD
`reference::load`; 99; this tree pins the key with primary
`$fpr` since $date, and a build made
before the signing ceremony pins none and refuses every reference, see
NEW
# The status lines keep the transparency log: it is pinned apart, later.
swap('docs/cloud.md', <<'OLD', <<"NEW");
> **Status: a session does not start from this source tree.** Its relay
> pins no archive key, so it refuses every image reference and exits 99
> (measured on 2026-10-24), and it pins no transparency log, which a relay
> with the key refuses with 100. `up` asks the relay first and stops there,
OLD
> **Status: a session does not start from this source tree.** Its relay
> pins the x0ne archive key, primary `$fpr`, since $date,
> but it pins no transparency log, so it refuses every image reference
> with 100. `up` asks the relay first and stops there,
NEW
swap('docs/attested-channel.md', <<'OLD', <<"NEW");
> **Status: does not start from this source tree.** The relay built here
> pins no archive key, so it refuses every image reference and exits 99
> (measured on 2026-10-24); with the key it would refuse them with 100, as
> it pins no transparency log. No `x0ne-cvm` image is published, and
OLD
> **Status: does not start from this source tree.** The relay built here
> pins the x0ne archive key, primary `$fpr`, since $date;
> it refuses every image reference with 100, as
> it pins no transparency log. No `x0ne-cvm` image is published, and
NEW
swap('README.md', <<'OLD', <<"NEW");
confidential**, and the cloud path does not start from this source tree: its
relay pins no archive key (exit 99) and no transparency log (exit 100), and no
image is published.
OLD
confidential**, and the cloud path does not start from this source tree: its
relay pins the x0ne archive key with primary `$fpr` since $date,
but no transparency log (exit 100), and no image is published.
NEW

# Write: the keyring, then each file through a temporary and a rename.
mkdir "$root/x0ne-infer/archive-key" or refuse("x0ne-infer/archive-key: $!")
    unless -d "$root/x0ne-infer/archive-key";
$new{$ring} = $k;
for my $rel (sort keys %new) {
    my $t = "$root/$rel.pin-tmp";
    open my $h, '>:raw', $t or die "$t: $!";
    print $h $new{$rel} or die "$t: $!";
    close $h or die "$t: $!";
    chmod((stat "$root/$rel")[2] & 07777, $t) if -e "$root/$rel";
    rename $t, "$root/$rel" or die "$rel: $!";
    print "pinned: $rel\n";
}
print "pin-archive-key: ALLOW_REHEARSAL_KEY=1: REHEARSAL key pinned, this tree is a rehearsal, never a release\n" if $rehearsal;
print "pin-archive-key: primary $fpr, user ID @uids, $date. Next: make ci, git diff, commit.\n";
PERL
