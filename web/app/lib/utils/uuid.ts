/**
 * Generate a UUID v4 string.
 *
 * Uses the native crypto.randomUUID() API available in all modern browsers.
 *
 * @returns UUID v4 string
 */
export function generateUUID(): string {
    return crypto.randomUUID()
}
