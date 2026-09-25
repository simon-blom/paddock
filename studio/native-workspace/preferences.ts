// Web and embedded viewers share one SQLite preference store. No WebKit copy.
import { initializeStudioPersistence } from '@/lib/browser-storage-migration'
import { uiPreferences, installPreferenceLifecycle } from '@/lib/ui-preferences'
export async function restorePreferences(): Promise<void> {
  await initializeStudioPersistence()
  installPreferenceLifecycle()
}
export const savePreferences = () => uiPreferences.flush()
