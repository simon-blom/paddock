<script setup lang="ts">
// Studio settings = only what changes how conversations behave.
// Everything else was evicted: tools/search are model config (the Manager's
// Start/Edit page), the DB export is manager admin (/manage/settings), theme
// lives in the header, and model facts live on the model's page.
import { computed, ref } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { useAudioDevices } from '@/composables/useAudioDevices'
import { OSM_TILES } from '@/lib/maptiles'
import ReplyLimitControl from './ReplyLimitControl.vue'
import Switch from '@/components/ui/Switch.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import TextInput from '@/components/ui/TextInput.vue'
import { TOOL_CALL_STOPS } from '@/lib/studio-settings-layout'

const settings = useSettingsStore()

// How many tools one reply may run. "Server default" (0 here, null on the
// wire) sends nothing and leaves the server's own budget alone - the
// honour-the-defaults stance. A number rides as the Responses API's
// `max_tool_calls`, which caps the calls AND lifts the server's round ceiling
// to match, so raising it actually buys more work rather than only less.
const toolCalls = computed<number>({
  get: () => settings.maxToolCalls ?? 0,
  set: (v) => {
    settings.maxToolCalls = v > 0 ? v : null
  },
})

// ── Microphone ──────────────────────────────────────────────────────────────
// The durable default for every mic path in the Studio. The COMPOSER's mic
// menu holds the same preference, because that is where the decision is
// actually made - you notice the wrong device with the mic about to open, not
// while reading a settings page. This is the other half: where you set the box
// up once, and the only place that can offer the permission step below.
//
// Output is deliberately absent. Routing playback needs `setSinkId`, which
// only Chromium implements, so a picker here would be a control that looks
// like it works and silently does not for everyone else - and the OS already
// routes output perfectly well.
const audio = useAudioDevices()
/** Reka reads '' as "nothing selected", so the system default needs a value of
 *  its own. Stored as an empty id, which is what sends no device constraint. */
const SYSTEM = 'system'
const micChoice = computed<string>({
  get: () => settings.micDeviceId || SYSTEM,
  set: (v) => {
    const hit = audio.devices.value.find((d) => d.id === v)
    settings.micDeviceId = v === SYSTEM ? '' : v
    // The label is stored so an UNPLUGGED device can still be named - a
    // disconnected one is not in `enumerateDevices` at all, and "your Jabra
    // headset isn't here" is actionable where "the microphone you chose isn't
    // here" is a riddle.
    settings.micDeviceLabel = v === SYSTEM ? '' : (hit?.label ?? '')
  },
})
const micOptions = computed<SelectOption[]>(() => {
  const opts: SelectOption[] = [{ value: SYSTEM, label: 'System default' }]
  for (const d of audio.devices.value) opts.push({ value: d.id, label: d.label })
  // The chosen device, when it is not plugged in right now. Listed rather than
  // dropped: removing it would move the selection onto a device nobody picked.
  if (settings.micDeviceId && audio.missing(settings.micDeviceId)) {
    opts.push({
      value: settings.micDeviceId,
      label: settings.micDeviceLabel || 'Chosen microphone',
      hint: 'not connected',
    })
  }
  return opts
})
const revealing = ref(false)
/** The permission prompt was dismissed or the microphone is blocked for this
 *  page. Said out loud: a button that quietly does nothing when clicked is the
 *  worst answer available, and "allow it and try again" is actionable. */
const revealDenied = ref(false)
// ── Map tiles ───────────────────────────────────────────────────────────────
// It belongs on this page by the rule above: it changes what a
// conversation does - whether opening a photo you attached makes a network
// request, and to whom. The head names the host rather than echoing the
// template, because the host is the part that matters here.

async function revealMics(): Promise<void> {
  revealing.value = true
  revealDenied.value = false
  try {
    revealDenied.value = !(await audio.reveal())
  } finally {
    revealing.value = false
  }
}

</script>

<template>
  <div class="settings">
    <h1 class="settings__title">Settings</h1>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Reply limit</h2>
        <ReplyLimitControl v-model="settings.maxTokens" class="settings__control" />
      </div>
      <p class="settings__sub">
        Applies to text replies, including thinking. Automatic uses each model’s available capacity.
        A custom value is an upper limit, not a target length.
      </p>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Tools per reply</h2>
        <Select v-model="toolCalls" :options="TOOL_CALL_STOPS" class="settings__pick" />
      </div>
      <p class="settings__sub">
        How many tools one reply may run before it answers with what it found. Reaching the limit
        is not an error - the reply says so and finishes.
      </p>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Summarize older messages</h2>
        <Switch v-model="settings.summarize" label="Summarize older messages" />
      </div>
      <p class="settings__sub">
        When a chat outgrows the context window, older messages are summarized in the background
        so the model keeps the thread of the conversation. Turn off to drop the oldest messages
        instead.
      </p>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Microphone</h2>
        <Select
          v-if="audio.named()"
          v-model="micChoice"
          :options="micOptions"
          class="settings__pick"
        />
        <button
          v-else-if="audio.supported()"
          class="pk-btn pk-btn--sm"
          type="button"
          :disabled="revealing"
          @click="revealMics"
        >
          Show my microphones
        </button>
      </div>
      <p v-if="!audio.supported()" class="settings__warn">
        The browser blocks the microphone on this address. See
        <RouterLink :to="{ name: 'trust' }">Trust this computer</RouterLink> in the Manager.
      </p>
      <p v-else-if="revealDenied" class="settings__warn">
        The microphone was blocked. Allow it for this page and try again.
      </p>
      <p v-else-if="!audio.named()" class="settings__sub">
        The browser only names your microphones once this page has been allowed to use one.
      </p>
      <p v-else-if="settings.micDeviceId && audio.missing(settings.micDeviceId)" class="settings__warn">
        {{ settings.micDeviceLabel || 'The microphone you chose' }} isn't connected. Recording uses
        the system default until it is back.
      </p>
    </section>

    <section class="settings__card">
      <div class="settings__head">
        <h2>Map tiles</h2>
        <TextInput v-model="settings.mapTiles" block placeholder="Follow the theme" aria-label="Map tiles" />
      </div>
      <p class="settings__sub">
        A photo with GPS shows a map drawn from an outline inside Paddock, which contacts nobody.
        Opening the interactive map fetches tiles from this address, which tells that host where
        the photo was taken. Leave it empty for a basemap that follows your theme, or name a
        server - your own, or OpenStreetMap's at {{ OSM_TILES }}.
      </p>
    </section>
  </div>
</template>

<style scoped>
.settings {
  container-type: inline-size;
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
}
.settings__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 20px;
}
.settings__card {
  display: grid;
  grid-template-columns: minmax(140px, 0.9fr) minmax(0, 1.5fr);
  gap: 8px 24px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 18px 20px;
  margin-bottom: 14px;
}
.settings__head {
  display: contents;
}
.settings__head h2 {
  grid-column: 1;
  padding-top: 7px;
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.settings__sub {
  grid-column: 2;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
  margin: 0;
}
.settings__warn {
  grid-column: 2;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-status-warning);
  line-height: 1.5;
  margin: 6px 0 0;
}
.settings__pick {
  min-width: 0;
  width: 100%;
}
.settings__head > :not(h2) { grid-column: 2; justify-self: start; max-width: 100%; }
.settings__control { width: 100%; }
@container (max-width: 520px) {
  .settings__card { grid-template-columns: minmax(0, 1fr); gap: 10px; }
  .settings__head > :not(h2), .settings__sub, .settings__warn { grid-column: 1; }
  .settings__head h2 { padding-top: 0; }
}
</style>
