/**
 * PDF text extraction via pdf.js, loaded lazily from CDN.
 *
 * Only fetched when a user actually attaches a PDF — zero bundle cost
 * otherwise.  Falls back gracefully if the CDN is unreachable (offline,
 * air-gapped network, etc.).
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

const PDFJS_CDN =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
const PDFJS_WORKER_CDN =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";

let pdfjsLib: any | null = null;

async function loadPdfJs(): Promise<any> {
  if (pdfjsLib) return pdfjsLib;
  try {
    pdfjsLib = await import(/* @vite-ignore */ PDFJS_CDN);
    pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_CDN;
    return pdfjsLib;
  } catch {
    throw new Error(
      "Could not load PDF.js. Check your network connection.",
    );
  }
}

export type PdfExtractionResult = {
  /** Concatenated text across all pages. */
  text: string;
  /** Number of pages in the PDF. */
  pageCount: number;
  /** Number of pages that yielded meaningful text. */
  pagesWithText: number;
  /** Approximate word count. */
  wordCount: number;
};

/**
 * Extract text from a PDF provided as an ArrayBuffer.
 *
 * Uses pdf.js to read every page's text content and concatenates it with
 * page markers so the LLM can reference "page N".
 */
export async function extractPdfText(
  buffer: ArrayBuffer,
): Promise<PdfExtractionResult> {
  const lib = await loadPdfJs();
  const doc = await lib.getDocument({ data: new Uint8Array(buffer) }).promise;
  const pageCount: number = doc.numPages;

  const pages: string[] = [];
  let pagesWithText = 0;

  for (let i = 1; i <= pageCount; i++) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    const items: Array<{ str: string; hasEOL?: boolean }> = content.items;

    // Join text items, respecting line breaks the PDF declares.
    let pageText = "";
    for (const item of items) {
      pageText += item.str;
      if (item.hasEOL) pageText += "\n";
    }

    const trimmed = pageText.trim();
    if (trimmed.length > 0) {
      pagesWithText++;
      pages.push(`--- Page ${i} ---\n${trimmed}`);
    }
  }

  const text = pages.join("\n\n");
  const wordCount = text.split(/\s+/).filter(Boolean).length;

  return { text, pageCount, pagesWithText, wordCount };
}

/**
 * Render PDF pages as JPEG images.
 *
 * Useful for scanned PDFs that have no extractable text — if a vision
 * model is available we can send the pages as images instead.
 *
 * Returns an array of data URLs (image/jpeg), one per page.
 */
export async function renderPdfPagesToImages(
  buffer: ArrayBuffer,
  opts?: { maxPages?: number; scale?: number; quality?: number },
): Promise<string[]> {
  const lib = await loadPdfJs();
  const doc = await lib.getDocument({ data: new Uint8Array(buffer) }).promise;
  const pageCount: number = doc.numPages;
  const maxPages = opts?.maxPages ?? 10;
  const scale = opts?.scale ?? 1.5; // ~150 DPI for typical PDFs
  const quality = opts?.quality ?? 0.85;

  const images: string[] = [];

  for (let i = 1; i <= Math.min(pageCount, maxPages); i++) {
    const page = await doc.getPage(i);
    const viewport = page.getViewport({ scale });
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    await page.render({ canvasContext: ctx, viewport }).promise;
    images.push(canvas.toDataURL("image/jpeg", quality));
  }

  return images;
}

/**
 * True if the given MIME type is a PDF.
 */
export function isPdfMimeType(mimeType: string): boolean {
  return mimeType === "application/pdf" || mimeType === "application/x-pdf";
}

/**
 * Convert a data URL to an ArrayBuffer.
 */
export function dataUrlToArrayBuffer(dataUrl: string): ArrayBuffer {
  const base64 = dataUrl.split(",")[1];
  if (!base64) throw new Error("Invalid data URL");
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}
