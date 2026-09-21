#ifndef PADDOCK_DESKTOP_H
#define PADDOCK_DESKTOP_H
#include <stdint.h>
// Private, versioned ABI. A core must be used and closed on one serial worker,
// never the main thread. Strings are UTF-8, owned by Rust; free them here only.
// open/snapshot return NULL on failure and set *error (caller initializes NULL).
// No pointers or strings may be retained after close/free. No concurrent close.
uint32_t paddock_desktop_abi_version(void);
void *paddock_desktop_open(char **error);
char *paddock_desktop_snapshot(void *core, char **error);
// ABI 11: read-only model defaults and full Start/Edit settings parity.
// ABI 10 added read-only open/poll/close runner-log subscriptions; <=1 KiB input.
// Bounded queues, background file I/O, credential suppression, no arbitrary paths.
char *paddock_desktop_logs(void *core, const uint8_t *bytes, uintptr_t len, char **error);
// ABI 8: typed asynchronous connector/search management; <=128 KiB, input-only secrets.
char *paddock_desktop_integrations(void *core, const uint8_t *bytes, uintptr_t len, char **error);
// ABI 7: typed connection review/check/save/poll commands, <=128 KiB.
// Keys are input-only; checks/saves run off the ABI queue and return receipts.
char *paddock_desktop_connections(void *core, const uint8_t *bytes, uintptr_t len, char **error);
// ABI 6: typed list/plan/pull/pause/resume, <=16 KiB; catalog IDs only.
char *paddock_desktop_downloads(void *core, const uint8_t *bytes, uintptr_t len, char **error);
// ABI 5: open the app-private full Studio host, input {assets: bundled path}.
// Result session goes into a native HttpOnly cookie, never JS or logs. <=8 KiB.
char *paddock_desktop_studio(void *core, const uint8_t *bytes, uintptr_t len, char **error);
// ABI 4: typed list/load/send/poll/cancel chat commands, input <=256 KiB.
// Poll is nonblocking and bounded; generations run on Rust workers. No keys or
// caller-chosen URLs. Durable conversations use the same store as browser Studio.
char *paddock_desktop_chat(void *core, const uint8_t *bytes, uintptr_t len, char **error);
// Typed JSON (1..65536 bytes). prepare returns credential-free unsaved settings;
// mutations return a job receipt immediately. Poll jobs in
// snapshot for completion. ABI 9 adds revision/PID-guarded edits and removal.
// Replacement keys are input-only; saved keys and raw TOML never return.
char *paddock_desktop_submit(void *core, const uint8_t *bytes, uintptr_t len, char **error);
// Anonymous OpenRouter catalog only. No core handle or user state is needed.
// Typed JSON (1..1024 bytes); bounded 45s/8MiB. Call on a separate background
// worker. Concurrent calls are supported (at most four in flight).
char *paddock_desktop_browse(const uint8_t *bytes, uintptr_t len, char **error);
void paddock_desktop_string_free(char *value);
void paddock_desktop_close(void *core);
// close waits for any accepted lifecycle transaction to settle. Never call
// on the UI thread; stopping healthy endpoints remains an explicit user action.
#endif
