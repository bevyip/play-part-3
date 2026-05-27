import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const appPath = path.join(__dirname, '..', 'app.js');
const dataPath = path.join(__dirname, 'suggestions-data.mjs');

const appJs = fs.readFileSync(appPath, 'utf8');
const raw = fs.readFileSync(dataPath, 'utf8');
const match = raw.match(/export const SUGGESTIONS\s*=\s*([\s\S]*);\s*$/);
if (!match) {
  console.error('Could not parse SUGGESTIONS from suggestions-data.mjs');
  process.exit(1);
}

const startMarker = 'const SUGGESTIONS = {';
const endMarker = 'const HERO_BG_GRADIENT';
const start = appJs.indexOf(startMarker);
const end = appJs.indexOf(endMarker, start);
if (start === -1 || end === -1) {
  console.error('Markers not found in app.js', { start, end });
  process.exit(1);
}

const replacement = `const SUGGESTIONS = ${match[1]};\n\n`;
const next = appJs.slice(0, start) + replacement + appJs.slice(end);
fs.writeFileSync(appPath, next);
console.log('Patched SUGGESTIONS in app.js');
