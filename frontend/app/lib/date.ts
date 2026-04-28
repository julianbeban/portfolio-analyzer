/**
 * Returns today's date as YYYY-MM-DD in the user's local timezone.
 * Use this instead of new Date().toISOString().split('T')[0] which returns UTC.
 */
export function getLocalDateString(date: Date = new Date()): string {
  return date.toLocaleDateString('en-CA'); // en-CA locale produces YYYY-MM-DD format
}