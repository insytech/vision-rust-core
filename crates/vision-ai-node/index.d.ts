/**
 * vision-ai-node - Rust optimizations for vision-ai Node.js server
 */

/**
 * Parse JSON string using SIMD acceleration (3-5x faster than JSON.parse)
 * @param jsonString - JSON string to parse
 * @returns Parsed JavaScript value
 */
export function jsonParse(jsonString: string): unknown;

/**
 * Stringify value to JSON (faster for large objects)
 * @param value - Value to stringify
 * @param pretty - Pretty print with indentation (default: false)
 * @returns JSON string
 */
export function jsonStringify(value: unknown, pretty?: boolean): string;

/**
 * Parse JSON array with SIMD - optimized for arrays of objects
 * @param jsonString - JSON array string
 * @returns Parsed array
 */
export function jsonParseArray<T = unknown>(jsonString: string): T[];

/**
 * Create ZIP from a directory (4-6x faster than archiver)
 * @param sourceDir - Directory to compress
 * @param zipPath - Output ZIP file path
 * @param compressionLevel - Compression level 0-9 (default: 6)
 * @returns Number of files compressed
 */
export function createZip(
  sourceDir: string,
  zipPath: string,
  compressionLevel?: number
): number;

/**
 * Extract ZIP to directory
 * @param zipPath - ZIP file to extract
 * @param destDir - Destination directory
 * @returns Array of extracted file paths
 */
export function extractZip(zipPath: string, destDir: string): string[];

/**
 * Create ZIP from specific file list
 * @param files - Array of file objects with path and name
 * @param zipPath - Output ZIP path
 * @param compressionLevel - Compression level (default: 6)
 * @returns Number of files added
 */
export function createZipFromFiles(
  files: Array<{ path: string; name: string }>,
  zipPath: string,
  compressionLevel?: number
): number;

/**
 * Transform revision data for API response
 * @param revision - Revision object with Class and File arrays
 * @returns Transformed revision object
 */
export function transformRevision(revision: {
  id: string;
  name?: string;
  status?: string;
  Class?: unknown[];
  File?: unknown[];
}): {
  id: string;
  name?: string;
  status?: string;
  classes: unknown[];
  files: unknown[];
};

/**
 * Transform files array (batch operation)
 * @param files - Array of file objects
 * @returns Transformed files array
 */
export function transformFiles(
  files: Array<{
    id: string;
    name: string;
    url: string;
    fileType?: string;
    thumbnailUrl?: string;
  }>
): Array<{
  id: string;
  name: string;
  url: string;
  fileType?: string;
  thumbnailUrl?: string;
}>;

/**
 * Transform classes array (batch operation)
 * @param classes - Array of class objects
 * @returns Transformed classes array
 */
export function transformClasses(
  classes: Array<{
    id: string;
    name: string;
    color?: string;
    index?: string | number;
  }>
): Array<{
  id: string;
  name: string;
  color?: string;
  index: number;
}>;
