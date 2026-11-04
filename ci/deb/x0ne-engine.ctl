Section: misc
Priority: optional
Standards-Version: 4.7.0
Package: x0ne-engine
Version: 0.2.0~beta.7
Maintainer: x0ne <dev@x0ne.co>
Architecture: amd64
Depends: libc6 (>= 2.36), libstdc++6, libgcc-s1, ca-certificates
Suggests: x0ne-gcloud-repo
Files: /out/bin/x0ne-engine /usr/bin/
 /out/bin/x0ne-engine-cloud /usr/bin/
 /out/bin/x0ne-engine-proxy /usr/bin/
 /out/bin/x0ne-engine.version /usr/share/x0ne-engine/
Description: x0ne Engine client: verified models and cloud sessions
 x0ne-engine pulls open models from a pinned catalog (every file verified by
 sha256), lists, verifies and removes them. x0ne-engine cloud runs
 x0ne-engine-cloud: one confidential GPU VM per session, its proof checked
 on this machine before any prompt leaves it, the questions carried by
 x0ne-engine-proxy over the attested channel (a cloud account and
 google-cloud-cli from x0ne-gcloud-repo are the user's). The attested
 server, x0ne-infer, runs in that VM and ships in its image only:
 nothing here installs it, and no server runs on this machine
 (x0ne-engine run is refused and points to x0ne-engine cloud run).
 Built from the x0ne-Engine commit written in
 /usr/share/x0ne-engine/x0ne-engine.version.
