import { defineConfig, mergeConfig } from 'vitest/config'
import studio from './vite.config.ts'

// Keep Vue/alias transforms, but do not execute the lab's node:test suites
// through Vitest (they have their own `node --test renderer-lab/*.test.mjs`).
export default mergeConfig(studio, defineConfig({ test: { include: ['src/**/*.test.ts'] } }))
