export function cleanToken(token: string): string {
  return token.replace(/^Ġ/, '');
}
export function displayToken(token: string): string {
  return token.replace(/^Ġ/, '').replace('<', '‹').replace('>', '›');
}