// A local Tessera log (github.com/transparency-dev/tessera, the POSIX
// storage driver: the library Rekor v2 is built on) holding x0ne image
// reference entries, for tlog_interop.rs. Local files only, no witness
// network: the witness cosignature is added here with
// transparency-dev/formats (cosignature/v1). Keys are derived from public
// strings: anyone can sign with them.
//
//	tessera make <dir>   300 references, checkpoints at 100 and 300, proofs
package main

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	fnote "github.com/transparency-dev/formats/note"
	"github.com/transparency-dev/tessera"
	"github.com/transparency-dev/tessera/client"
	"github.com/transparency-dev/tessera/storage/posix"
	"golang.org/x/mod/sumdb/note"
)

const (
	origin    = "interop.invalid/tessera-log"
	witness   = "interop.invalid/tessera-witness"
	entryHead = "x0ne-engine/image-reference/v1"
	proofHead = "c2sp.org/tlog-proof@v1"
	refIndex  = 42
	size1     = 100
	size2     = 300
)

func die(f string, a ...any) { fmt.Fprintf(os.Stderr, "FAIL: "+f+"\n", a...); os.Exit(1) }

func seed(label string) []byte { s := sha256.Sum256([]byte("x0ne-engine tessera interop, public: " + label)); return s[:] }

func keyHash(name string, algKey []byte) uint32 {
	h := sha256.New()
	h.Write([]byte(name))
	h.Write([]byte("\n"))
	h.Write(algKey)
	return binary.BigEndian.Uint32(h.Sum(nil)[:4])
}

// keys: an Ed25519 signer key (type 0x01, as formats' cosigner reads it too)
// and the verifier key of type alg (0x01 note signature, 0x04 cosignature/v1).
func keys(name string, alg byte, sd []byte) (skey, vkey string) {
	pub := ed25519.NewKeyFromSeed(sd).Public().(ed25519.PublicKey)
	k1 := append([]byte{0x01}, pub...)
	skey = fmt.Sprintf("PRIVATE+KEY+%s+%08x+%s", name, keyHash(name, k1), base64.StdEncoding.EncodeToString(append([]byte{0x01}, sd...)))
	k := append([]byte{alg}, pub...)
	vkey = fmt.Sprintf("%s+%08x+%s", name, keyHash(name, k), base64.StdEncoding.EncodeToString(k))
	return
}

func entry(ref []byte) []byte {
	d := sha256.Sum256(ref)
	return []byte(fmt.Sprintf("%s\nsha256 %s\n", entryHead, hex.EncodeToString(d[:])))
}

func write(path string, b []byte) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		die("%v", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		die("%v", err)
	}
}

func b64lines(hs [][]byte) string {
	var s strings.Builder
	for _, h := range hs {
		s.WriteString(base64.StdEncoding.EncodeToString(h) + "\n")
	}
	return s.String()
}

func main() {
	if len(os.Args) != 3 || os.Args[1] != "make" {
		die("usage: tessera make <dir>")
	}
	dir := os.Args[2]
	ctx := context.Background()
	logSkey, logVkey := keys(origin, 0x01, seed("log"))
	witSkey, witVkey := keys(witness, 0x04, seed("witness"))
	logSigner, err := note.NewSigner(logSkey)
	if err != nil {
		die("log signer: %v", err)
	}
	logVerifier, err := note.NewVerifier(logVkey)
	if err != nil {
		die("log verifier: %v", err)
	}
	cosigner, err := fnote.NewSignerForCosignatureV1(witSkey)
	if err != nil {
		die("cosigner: %v", err)
	}
	store := filepath.Join(dir, "store")
	driver, err := posix.New(ctx, posix.Config{Path: store})
	if err != nil {
		die("posix: %v", err)
	}
	appender, shutdown, reader, err := tessera.NewAppender(ctx, driver, tessera.NewAppendOptions().
		WithCheckpointSigner(logSigner).
		WithBatching(size1, 100*time.Millisecond).
		WithCheckpointInterval(100*time.Millisecond))
	if err != nil {
		die("appender: %v", err)
	}
	await := tessera.NewPublicationAwaiter(ctx, reader.ReadCheckpoint, 50*time.Millisecond)
	// add appends references [from, to) and waits for a checkpoint of size `to`.
	add := func(from, to int) []byte {
		for i := from; i < to; i++ {
			ref := []byte(fmt.Sprintf("{\"tessera\": %d}\n", i))
			write(filepath.Join(dir, fmt.Sprintf("r-%03d.json", i)), ref)
			idx, _, err := await.Await(ctx, appender.Add(ctx, tessera.NewEntry(entry(ref))))
			if err != nil {
				die("add %d: %v", i, err)
			}
			if idx.Index != uint64(i) {
				die("entry %d sequenced at %d", i, idx.Index)
			}
		}
		for {
			cp, err := reader.ReadCheckpoint(ctx)
			if err == nil {
				n, err := note.Open(cp, note.VerifierList(logVerifier))
				if err != nil {
					die("tessera checkpoint: %v", err)
				}
				if strings.Split(n.Text, "\n")[1] == fmt.Sprint(to) {
					// The witness cosigns what the log signed.
					signed, err := note.Sign(n, cosigner)
					if err != nil {
						die("cosign: %v", err)
					}
					return signed
				}
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
	tiles := client.TileFetcherFunc(reader.ReadTile)
	cp1 := add(0, size1)
	write(filepath.Join(dir, fmt.Sprintf("checkpoint-%d", size1)), cp1)
	pb, err := client.NewProofBuilder(ctx, size1, tiles)
	if err != nil {
		die("proof builder: %v", err)
	}
	ip, err := pb.InclusionProof(ctx, refIndex)
	if err != nil {
		die("inclusion: %v", err)
	}
	proof := fmt.Sprintf("%s\nindex %d\n%s\n%s", proofHead, refIndex, b64lines(ip), cp1)
	write(filepath.Join(dir, fmt.Sprintf("r-%03d.json.tlog-proof", refIndex)), []byte(proof))
	cp2 := add(size1, size2)
	pb, err = client.NewProofBuilder(ctx, size2, tiles)
	if err != nil {
		die("proof builder: %v", err)
	}
	cons, err := pb.ConsistencyProof(ctx, size1, size2)
	if err != nil {
		die("consistency: %v", err)
	}
	write(filepath.Join(dir, fmt.Sprintf("consistency-%d-%d", size1, size2)), []byte(b64lines(cons)))
	if err := shutdown(ctx); err != nil {
		die("shutdown: %v", err)
	}
	// The cosigned checkpoint replaces the log's own once the appender is down.
	write(filepath.Join(store, "checkpoint"), cp2)
	write(filepath.Join(dir, "vkeys"), []byte(fmt.Sprintf("log %s\nwitness %s\n", logVkey, witVkey)))
	fmt.Printf("tessera: %d entries, checkpoints %d and %d, proof of %d, consistency %d to %d\n", size2, size1, size2, refIndex, size1, size2)
}
