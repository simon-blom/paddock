# macOS releases from the public repository

This pipeline builds **`truespar/paddock`** and refuses any other origin. It creates
the native app and standalone command-line tools from the same source snapshot.
It never installs an app, changes the user's library, starts a model, commits,
pushes, creates a GitHub release, or uploads to the model bucket.

The first distribution target is **Apple Silicon, macOS 26 or later**. The
Swift UI's development deployment target remains macOS 15, but the bundled
runner compiles Metal Shading Language 4.0. The release build explicitly sets
26.0 for both Rust and Swift and the installer, rather than inheriting the build
machine's OS. This is not an Intel build or a claim of qualification on every
Apple GPU. Hardware/model qualification remains a separate release gate.

## Outputs

- `Paddock-<version>-macos-arm64.dmg`: drag `Paddock.app` to Applications. The
  app contains its Rust management library and Metal runner; no separate manager,
  Python, Node, Rust or Swift installation is required on the user's Mac.
- `Paddock-<version>-macos-arm64-cli.pkg`: installs `paddock` and
  `paddock-runner` under `/usr/local/bin`, with licenses under
  `/usr/local/share/paddock`. No background service or automatic model start.
- `Paddock-<version>-macos-arm64-cli.tar.gz`: the same command-line payload for
  manual installation. Notarization tickets cannot be stapled to bare binaries
  or tar archives; use the stapled PKG for offline Gatekeeper validation.
- `SHA256SUMS`, source/lockfile/toolchain provenance, and notarization records.

Weights, user databases, saved credentials and the optional private catalog are
never release inputs. The native app packages the viewer-only web build for
Lector, Scriptor and Traverse. It does not package the diagnostic web transcript.
The standalone manager separately embeds the full web Studio.

## Build prerequisites

Use an Apple Silicon Mac with a production Xcode selected by `xcode-select`;
`apps/macos/scripts/check-toolchain.sh` enforces the existing Swift baseline.
Install the repository's stable Rust toolchain, Node/npm, CMake (for the native
audio codec build), and Python 3.9+. CMake must be on the build command's PATH;
an unpacked [official binary](https://cmake.org/download/) is sufficient.
No third-party Python packages or global toolchain changes are needed.
Allow at least 40 GiB for a cold build. Keep the build host patched.

The dependency lockfiles are authoritative. The pipeline uses `npm ci`, Cargo
`--locked`, SwiftPM `--force-resolved-versions`, and checks that all three
lockfiles remain unchanged. It records exact compiler and SDK versions; it does
not claim bit-for-bit reproducibility across compiler versions or signing times.

From the public repository root:

```sh
python3 -B -m unittest discover -s packaging/macos -p 'test_*.py'
python3 packaging/macos/release.py doctor
python3 packaging/macos/release.py build \
  --output dist/macos-candidate-001 --build-number 1
```

Choose a **new output directory** each time. Builds run in an isolated snapshot
under that directory, not against another checkout's generated files. Every
Rust component and the shipping Swift product is optimized. Metal is enabled;
the runner uses the macOS CoreGraphics PDF path, not the Windows/Linux PDFium
archive. Manager, desktop core and runner use their `hardened` features.

The app marketing version comes from `[workspace.package].version` in
`Cargo.toml`. Supply a monotonically increasing numeric `--build-number`; Git
commit counts are not a stable app build sequence. Binaries retain their commit
stamp and the build manifest includes the full source hash inventory.

The app icon is the committed `Paddock.icns`, rendered from the Paddock mark
in `studio/public/img/paddock-mark.svg` by
`swift packaging/macos/make-icon.swift`, which also renders `assets/paddock.ico`
(the exes and the Studio favicon) and the pair for each alternative under
`assets/icon-alternatives/`. Rerun it and commit the results when a mark
changes; neither build script renders them. The menu bar item draws
`apps/macos/Sources/PaddockUI/Resources/PaddockMenuBar.svg` as a template image.

`build` uses ad-hoc signatures only and produces visibly `UNSIGNED` artifacts.
These are packaging candidates, **not customer downloads**. It checks native
viewer boundaries, arm64 dependencies and signatures, CLI startup, and a real
desktop ABI open/snapshot/close/reopen in a fresh temporary library. It does not
open the app UI or exercise existing credentials.

For changes not yet committed, add `--allow-dirty`. These candidates are always
ineligible for `finalize`. To distribute, land the reviewed changes publicly and
build again from a clean checkout. Merely renaming an unsigned artifact is not a
release process.

## One-time signing setup

Use the team's Apple Developer Program **Account Holder** to create two local
Developer ID identities. Do not choose Apple Development, Apple Distribution,
or the Mac App Store certificate types. Service certificates (APNs, Apple Pay,
Wallet, Swift Package signing) are not needed for this distribution.

1. Open **Keychain Access**, not Passwords. In the macOS menu bar choose
   **Keychain Access > Certificate Assistant > Request a Certificate from a
   Certificate Authority**.
2. Enter the account email and a name such as `Paddock Application Signing`.
   Leave CA Email Address blank, select Saved to disk, and save the CSR outside
   the repository.
3. In [Certificates, Identifiers & Profiles](https://developer.apple.com/account/resources/certificates/list),
   select the correct team and create **Developer ID Application** with that CSR.
   Download the `.cer` and double-click it to import into the login keychain.
4. Create a **new CSR** named `Paddock Installer Signing`, and repeat for
   **Developer ID Installer**. Apple rejects reusing the first CSR here.
5. In Keychain Access > My Certificates, each certificate must have its matching
   private key. `security find-identity -v` should list both as valid identities
   with the same team ID. A `.cer` alone on a different Mac is not sufficient.

Application signs the app, dylib and executables. Installer signs the CLI PKG.
Do not export private keys into the repo, logs or chat. Arrange an encrypted
backup of both identities through the team's credential process; a signing Mac
should not be the only recoverable copy.

[Apple's certificate guide](https://developer.apple.com/help/account/certificates/create-developer-id-certificates)
and [CSR guide](https://developer.apple.com/help/account/certificates/create-a-certificate-signing-request).

## One-time notarization setup

At [account.apple.com](https://account.apple.com/), sign into the developer Apple
Account, open Sign-In and Security > App-Specific Passwords, and generate
`Paddock Notarization`. Two-factor authentication is required. This is the Apple
Account site, not the certificate portal.

In your own Terminal, substitute the account email and team ID:

```sh
xcrun notarytool store-credentials "paddock-notary" \
  --apple-id "YOUR-DEVELOPER-EMAIL" --team-id "YOURTEAMID"
```

Paste the app-specific password only at the secure prompt. Omitting
`--password` keeps the secret out of shell history and process arguments. The
tool validates and saves it in Keychain. The build script receives only the
profile name. No password belongs in the repository or a `.env` file.

An App Store Connect API key is another supported authentication option if the
team later moves to unattended release automation. Do not create one just to
duplicate a working local notarization profile.

[Apple's app-specific password instructions](https://support.apple.com/en-us/102654).

## Sign and notarize a reviewed candidate

Use the **full identity names**, including team IDs, from `security find-identity -v`:

```sh
python3 packaging/macos/release.py finalize \
  --input dist/macos-candidate-001 \
  --application-identity "Developer ID Application: YOUR ORGANIZATION (YOURTEAMID)" \
  --installer-identity "Developer ID Installer: YOUR ORGANIZATION (YOURTEAMID)" \
  --notary-profile "paddock-notary"
```

The command refuses dirty or non-public source provenance, mismatched teams,
missing private keys and any changed candidate files. It works on another copy;
the unsigned candidate remains intact. New release outputs are never overwritten.

It signs nested code inside-out with secure timestamps and Hardened Runtime.
Only the app receives the microphone entitlement; no App Sandbox, JIT,
`get-task-allow` or disabled-library-validation exceptions are added. It submits
the app to Apple, fetches the notarization log, requires `Accepted`, and staples
the app before making the DMG. DMG and signed PKG are then separately notarized
and stapled, followed by signature, stapler and Gatekeeper checks. Checksums are
computed **after** stapling. Failed steps do not produce a successful manifest.

Apple can take time to process a first submission. Preserve the candidate,
submission IDs and logs if it fails; do not bypass rejection or re-sign files
after stapling. Investigate the log and build a fresh candidate after corrections.

[Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)
and [distribution signing](https://developer.apple.com/documentation/xcode/creating-distribution-signed-code-for-the-mac/).

## Final release gates

Notarization is a security check, not proof that the app works. The script marks
its signed output `notarized-awaiting-manual-qualification`. Before publishing:

- Run `apps/macos/scripts/check.sh` on the exact public commit. Its native visual
  checks need an unlocked desktop and can open temporary test windows.
- Test a browser-downloaded, quarantined DMG and PKG on a clean Mac/user account,
  including the declared minimum OS. Test offline launch after stapling.
- Verify native startup, first-run and denied microphone/notification permissions,
  Keychain storage/relaunch, downloads, text and image inference, PDF/DOCX/graph
  viewers, native audio, model start/stop and externally reachable runner APIs.
- Verify an update signed by the same team preserves conversations, models and
  credentials. Verify no new permissions or repeated credential prompts.
- Check both CLI binaries, bundle version/build number, model catalog, licenses,
  installed paths and checksums. Record the actual hardware/model coverage.
- Review release notes and approve the version/tag. Publish only the named
  artifacts and checksums, **not** work directories, source snapshots, component
  packages, notarization upload ZIPs or credentials.

Automatic in-app updates and a public macOS CI/release workflow are not added by
this script. Start with local, explicit releases. Never attach this personal Mac
as a self-hosted runner for untrusted public pull requests: those jobs would run
alongside the signing identities and developer data. Any later automation needs
an isolated, controlled release host and a separate trust boundary.
