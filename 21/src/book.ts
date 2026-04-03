import { layoutNextLine, prepareWithSegments, type LayoutCursor, type PreparedTextWithSegments } from '@chenglou/pretext'
import {
  BODY_TEXT_AFTER,
  BODY_TEXT_BEFORE,
  BOOK_TITLE,
  CHAPTER_LABEL,
  FIGURE_CAPTION_BODY,
  FIGURE_LABEL,
  SECTION_HEADING,
} from './content'

export type ToolId = 'pen' | 'marker' | 'brush' | 'eraser'

/** Optional `lw` = line width at this vertex (paintbrush dynamics). */
export type StrokePoint = { x: number; y: number; lw?: number }

export type Stroke = {
  tool: ToolId
  color: string
  width: number
  points: StrokePoint[]
}

type Column = { x: number; y: number; w: number; h: number }

const BODY_FONT = '400 16px "EB Garamond", "Libre Baskerville", Georgia, serif'
const CAPTION_FONT = 'italic 400 14px "EB Garamond", "Libre Baskerville", Georgia, serif'
const FIGURE_LABEL_FONT = 'italic 600 14px "EB Garamond", "Libre Baskerville", Georgia, serif'
const HEADING_FONT_SMALL = '600 11px "Cormorant SC", "EB Garamond", serif'
const TITLE_FONT = '700 21px "EB Garamond", "Libre Baskerville", Georgia, serif'
const LINE_HEIGHT = 24
const CAPTION_LINE_HEIGHT = 20
const MIN_LINE_WIDTH = 26

/** Second column (index 1): left page, inner column — figure row is vertically centered here. */
const FIGURE_COLUMN_INDEX = 1
const TEXT_MARGIN_X = 44
const TEXT_TOP = 88
const TEXT_BOTTOM_PAD = 52
/** Top of figure image: fixed inset from column top so doodles never shift the bitmap (they may draw on top). */
const FIGURE_TOP_INSET = 44
/** Space below image/caption before body text resumes in column 1 and neighbours. */
const FIGURE_AFTER_PAD = 12

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

/** Calligraphy stroke thins toward both ends; very short strokes stay near full width. */
function brushEndTaper(ptIndex: number, ptCount: number): number {
  if (ptCount <= 2) return 1
  const u =
    Math.min(ptIndex, ptCount - 1 - ptIndex) / Math.max((ptCount - 1) * 0.5, 0.001)
  return 0.64 + 0.36 * clamp(u * 1.12, 0, 1)
}

function brushSegmentCoreWidth(s: Stroke, segIndex: number): number {
  const pts = s.points
  const p0 = pts[segIndex]
  const p1 = pts[segIndex + 1]
  const w0 = p0.lw ?? s.width
  const w1 = p1.lw ?? s.width
  const base = (w0 + w1) / 2
  const t0 = brushEndTaper(segIndex, pts.length)
  const t1 = brushEndTaper(segIndex + 1, pts.length)
  return base * ((t0 + t1) / 2)
}

function findWidestRun(
  data: ImageData,
  y: number,
  xMin: number,
  xMax: number,
): { x: number; width: number } {
  const { width, height, data: buf } = data
  const yi = clamp(Math.round(y), 0, height - 1)
  const x0 = clamp(Math.round(xMin), 0, width - 1)
  const x1 = clamp(Math.round(xMax), 0, width)
  let bestStart = x0
  let bestLen = 0
  let curStart = -1
  for (let x = x0; x < x1; x++) {
    const i = (yi * width + x) * 4
    const free = buf[i] > 210 && buf[i + 1] > 210 && buf[i + 2] > 210
    if (free) {
      if (curStart < 0) curStart = x
    } else {
      if (curStart >= 0) {
        const len = x - curStart
        if (len > bestLen) {
          bestLen = len
          bestStart = curStart
        }
        curStart = -1
      }
    }
  }
  if (curStart >= 0) {
    const len = x1 - curStart
    if (len > bestLen) {
      bestLen = len
      bestStart = curStart
    }
  }
  return { x: bestStart, width: bestLen }
}

function rebuildMask(strokes: Stroke[], w: number, h: number): ImageData {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, w, h)
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  for (const s of strokes) {
    if (s.points.length === 0) continue
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1

    if (s.tool === 'brush') {
      const pad = 9
      if (s.points.length === 1) {
        const p = s.points[0]
        const w = (p.lw ?? s.width) + pad
        ctx.fillStyle = '#000000'
        ctx.beginPath()
        ctx.arc(p.x, p.y, w / 2, 0, Math.PI * 2)
        ctx.fill()
        continue
      }
      ctx.strokeStyle = '#000000'
      for (let i = 0; i < s.points.length - 1; i++) {
        const p0 = s.points[i]
        const p1 = s.points[i + 1]
        const lw = brushSegmentCoreWidth(s, i) + pad
        ctx.lineWidth = lw
        ctx.beginPath()
        ctx.moveTo(p0.x, p0.y)
        ctx.lineTo(p1.x, p1.y)
        ctx.stroke()
      }
      continue
    }

    const pad =
      s.tool === 'eraser' ? 6 : s.tool === 'marker' ? 16 : 10
    const lw = s.width + pad
    if (s.points.length === 1) {
      const p = s.points[0]
      ctx.fillStyle = s.tool === 'eraser' ? '#ffffff' : '#000000'
      ctx.beginPath()
      ctx.arc(p.x, p.y, lw / 2, 0, Math.PI * 2)
      ctx.fill()
      continue
    }
    ctx.strokeStyle = s.tool === 'eraser' ? '#ffffff' : '#000000'
    ctx.lineWidth = lw
    ctx.beginPath()
    ctx.moveTo(s.points[0].x, s.points[0].y)
    for (let i = 1; i < s.points.length; i++) {
      ctx.lineTo(s.points[i].x, s.points[i].y)
    }
    ctx.stroke()
  }
  return ctx.getImageData(0, 0, w, h)
}

function buildColumnsBand(spreadW: number, bandTop: number, bandHeight: number): Column[] {
  const marginX = TEXT_MARGIN_X
  const gutter = 40
  const colGap = 18
  const innerW = spreadW - marginX * 2
  const pageW = (innerW - gutter) / 2
  const colW = (pageW - colGap) / 2
  const leftX = marginX
  const rightPageX = marginX + pageW + gutter
  return [
    { x: leftX, y: bandTop, w: colW, h: bandHeight },
    { x: leftX + colW + colGap, y: bandTop, w: colW, h: bandHeight },
    { x: rightPageX, y: bandTop, w: colW, h: bandHeight },
    { x: rightPageX + colW + colGap, y: bandTop, w: colW, h: bandHeight },
  ]
}

const CAPTION_MIN_LINE_WIDTH = 22

function preparedExhausted(prepared: PreparedTextWithSegments, cursor: LayoutCursor): boolean {
  return (
    cursor.segmentIndex >= prepared.segments.length ||
    (cursor.segmentIndex === prepared.segments.length - 1 &&
      cursor.graphemeIndex >= (prepared.segments[cursor.segmentIndex]?.length ?? 0))
  )
}

/** Vertical band where body text must not intrude (figure column + caption spill columns). */
type ColumnBand = {
  columnIndex: number
  top: number
  bottom: number
}

function mergeMaxYPerColumn(into: Map<number, number>, from: Map<number, number>): void {
  for (const [k, v] of from) {
    into.set(k, Math.max(into.get(k) ?? 0, v))
  }
}

/**
 * Lay out one prepared caption/label block across columns. In the figure column, lines that still fit
 * wholly above `imgBottom` use the narrow strip beside the image; once lower, lines use full column
 * width (wrap under the image). Later columns always use full width.
 */
function flowCaptionPreparedMultiColumn(
  ctx: CanvasRenderingContext2D | null,
  mask: ImageData,
  prepared: PreparedTextWithSegments,
  fontShorthand: string,
  columns: Column[],
  startColumnIndex: number,
  yStart: number,
  figureColumnIndex: number,
  imgBottom: number,
  captionX: number,
  captionW: number,
): {
  yEnd: number
  endColumnIndex: number
  exhausted: boolean
  maxYPerColumn: Map<number, number>
} {
  let cursor: LayoutCursor = { segmentIndex: 0, graphemeIndex: 0 }
  if (ctx) {
    ctx.font = fontShorthand
  }
  const maxYPerColumn = new Map<number, number>()
  const recordLine = (ci: number, yTop: number) => {
    const lineBottom = yTop + CAPTION_LINE_HEIGHT
    maxYPerColumn.set(ci, Math.max(maxYPerColumn.get(ci) ?? 0, lineBottom))
  }
  const exhaustedPrep = (): boolean => preparedExhausted(prepared, cursor)

  let col = clamp(startColumnIndex, 0, Math.max(0, columns.length - 1))
  let y = yStart

  while (col < columns.length) {
    const colObj = columns[col]
    if (y < colObj.y) y = colObj.y
    const yClip = Math.min(colObj.y + colObj.h - 4, mask.height - 4)
    if (yClip < colObj.y + CAPTION_LINE_HEIGHT) {
      col += 1
      y = columns[col]?.y ?? y
      continue
    }

    while (y + CAPTION_LINE_HEIGHT <= yClip) {
      if (exhaustedPrep()) {
        return { yEnd: y, endColumnIndex: col, exhausted: true, maxYPerColumn }
      }
      const narrow = col === figureColumnIndex && y + CAPTION_LINE_HEIGHT <= imgBottom
      const x1 = narrow ? captionX : colObj.x
      const x2 = narrow ? captionX + captionW : colObj.x + colObj.w
      const sampleY = y + CAPTION_LINE_HEIGHT * 0.55
      const { x: runX, width: runW } = findWidestRun(mask, sampleY, x1, x2)
      if (runW < CAPTION_MIN_LINE_WIDTH) {
        y += CAPTION_LINE_HEIGHT
        continue
      }
      const line = layoutNextLine(prepared, cursor, runW)
      if (!line || !line.text.length) {
        y += CAPTION_LINE_HEIGHT
        continue
      }
      if (ctx) {
        ctx.save()
        const clipL = Math.max(0, colObj.x)
        const clipT = Math.max(0, colObj.y)
        const clipR = Math.min(mask.width, colObj.x + colObj.w)
        const clipB = Math.min(mask.height, colObj.y + colObj.h)
        if (clipR > clipL && clipB > clipT) {
          ctx.beginPath()
          ctx.rect(clipL, clipT, clipR - clipL, clipB - clipT)
          ctx.clip()
        }
        ctx.fillText(line.text, runX, y + CAPTION_LINE_HEIGHT * 0.75)
        ctx.restore()
      }
      cursor = line.end
      recordLine(col, y)
      y += CAPTION_LINE_HEIGHT
    }

    if (exhaustedPrep()) {
      return { yEnd: y, endColumnIndex: col, exhausted: true, maxYPerColumn }
    }
    col += 1
    if (col < columns.length) {
      y = columns[col].y
    }
  }

  return {
    yEnd: y,
    endColumnIndex: Math.max(0, columns.length - 1),
    exhausted: exhaustedPrep(),
    maxYPerColumn,
  }
}

function layoutFigureCaptionPretext(
  ctx: CanvasRenderingContext2D | null,
  mask: ImageData,
  labelPrep: PreparedTextWithSegments | null,
  bodyPrep: PreparedTextWithSegments | null,
  columns: Column[],
  figureColumnIndex: number,
  captionX: number,
  captionW: number,
  imgBottom: number,
  yStart: number,
): { y: number; complete: boolean; lastColumnIndex: number; maxYPerColumn: Map<number, number> } {
  const agg = new Map<number, number>()
  let col = figureColumnIndex
  let y = yStart
  let complete = true

  if (labelPrep) {
    const r = flowCaptionPreparedMultiColumn(
      ctx,
      mask,
      labelPrep,
      FIGURE_LABEL_FONT,
      columns,
      col,
      y,
      figureColumnIndex,
      imgBottom,
      captionX,
      captionW,
    )
    mergeMaxYPerColumn(agg, r.maxYPerColumn)
    if (!r.exhausted) complete = false
    y = r.yEnd + 4
    col = r.endColumnIndex
    const cobj = columns[col]
    if (cobj && y + CAPTION_LINE_HEIGHT > cobj.y + cobj.h - 4) {
      if (col + 1 < columns.length) {
        col += 1
        y = columns[col].y
      }
    }
  }

  if (bodyPrep) {
    const r = flowCaptionPreparedMultiColumn(
      ctx,
      mask,
      bodyPrep,
      CAPTION_FONT,
      columns,
      col,
      y,
      figureColumnIndex,
      imgBottom,
      captionX,
      captionW,
    )
    mergeMaxYPerColumn(agg, r.maxYPerColumn)
    if (!r.exhausted) complete = false
    y = r.yEnd
    col = r.endColumnIndex
  }

  return { y, complete, lastColumnIndex: col, maxYPerColumn: agg }
}

function buildCaptionSkipBands(
  columns: Column[],
  figureColumnIndex: number,
  slotTop: number,
  imgBottom: number,
  capTop: number,
  maxYPerColumn: Map<number, number>,
): ColumnBand[] {
  const bands: ColumnBand[] = []
  const col1 = columns[figureColumnIndex]
  const y1 = maxYPerColumn.get(figureColumnIndex)
  const bottom1 = Math.min(col1.y + col1.h, Math.max(imgBottom, y1 ?? capTop) + FIGURE_AFTER_PAD)
  if (bottom1 > slotTop + 2) {
    bands.push({ columnIndex: figureColumnIndex, top: slotTop, bottom: bottom1 })
  }
  for (let idx = figureColumnIndex + 1; idx < columns.length; idx++) {
    const ym = maxYPerColumn.get(idx)
    if (ym === undefined) continue
    const c = columns[idx]
    const bottom = Math.min(c.y + c.h, ym + FIGURE_AFTER_PAD)
    if (bottom > c.y + 2) {
      bands.push({ columnIndex: idx, top: c.y, bottom })
    }
  }
  return bands
}

function bandSkipTarget(ci: number, y: number, lineHeight: number, bands: ColumnBand[]): number | null {
  for (const b of bands) {
    if (b.columnIndex !== ci) continue
    if (y + lineHeight > b.top && y < b.bottom) {
      return b.bottom
    }
  }
  return null
}

type ColumnFlowState = {
  layoutCursor: LayoutCursor
  columnIndex: number
  /** Next line top (y) in `columnIndex` when resuming this prepared text. */
  y: number
}

type FlowResult = {
  exhausted: boolean
  contentBottomY: number
  state: ColumnFlowState
}

function flowTextResumable(
  ctx: CanvasRenderingContext2D,
  mask: ImageData,
  prepared: PreparedTextWithSegments,
  columns: Column[],
  layoutStart: LayoutCursor,
  spatialResume: { columnIndex: number; y: number } | null,
  captionBands: ColumnBand[] | null,
): FlowResult {
  let cursor: LayoutCursor = { ...layoutStart }
  const done = (): boolean =>
    cursor.segmentIndex >= prepared.segments.length ||
    (cursor.segmentIndex === prepared.segments.length - 1 &&
      cursor.graphemeIndex >= (prepared.segments[cursor.segmentIndex]?.length ?? 0))

  let ci = spatialResume?.columnIndex ?? 0
  let pendingSpatial: { columnIndex: number; y: number } | null = spatialResume

  const firstY = columns[0]?.y ?? TEXT_TOP
  let contentBottomY = firstY
  let drewLine = false
  let brokeBecauseDone = false
  let lastY = firstY

  const mw = mask.width
  const mh = mask.height

  ctx.save()
  ctx.font = BODY_FONT
  ctx.fillStyle = '#1a1a1a'

  while (ci < columns.length) {
    const col = columns[ci]
    let y: number
    if (pendingSpatial !== null && ci === pendingSpatial.columnIndex) {
      y = clamp(pendingSpatial.y, col.y, col.y + col.h)
      pendingSpatial = null
    } else {
      y = col.y
    }

    ctx.save()
    const clipLeft = Math.max(0, col.x)
    const clipTop = Math.max(0, col.y)
    const clipRight = Math.min(mw, col.x + col.w)
    const clipBottom = Math.min(mh, col.y + col.h)
    const clipW = clipRight - clipLeft
    const clipH = clipBottom - clipTop
    if (clipW > 0 && clipH > 0) {
      ctx.beginPath()
      ctx.rect(clipLeft, clipTop, clipW, clipH)
      ctx.clip()
    }

    while (y + LINE_HEIGHT <= col.y + col.h) {
      if (done()) {
        brokeBecauseDone = true
        break
      }
      const skipTo =
        captionBands !== null && captionBands.length > 0
          ? bandSkipTarget(ci, y, LINE_HEIGHT, captionBands)
          : null
      if (skipTo !== null) {
        y = Math.min(skipTo, col.y + col.h)
        continue
      }
      const sampleY = y + LINE_HEIGHT * 0.55
      const { x: runX, width: runW } = findWidestRun(mask, sampleY, col.x, col.x + col.w)
      if (runW < MIN_LINE_WIDTH) {
        y += LINE_HEIGHT
        continue
      }
      const line = layoutNextLine(prepared, cursor, runW)
      if (!line || !line.text.length) {
        y += LINE_HEIGHT
        continue
      }
      ctx.fillText(line.text, runX, y + LINE_HEIGHT * 0.78)
      cursor = line.end
      contentBottomY = Math.max(contentBottomY, y + LINE_HEIGHT)
      drewLine = true
      y += LINE_HEIGHT
    }

    ctx.restore()

    lastY = y

    if (brokeBecauseDone) break
    ci++
  }

  ctx.restore()

  if (!drewLine && spatialResume === null && layoutStart.segmentIndex === 0 && layoutStart.graphemeIndex === 0) {
    contentBottomY = firstY
  }

  let outCi = ci
  let outY = lastY
  if (outCi >= columns.length && columns.length > 0) {
    const lc = columns[columns.length - 1]
    outCi = columns.length - 1
    outY = lc.y + lc.h
  }

  return {
    exhausted: brokeBecauseDone || done(),
    contentBottomY,
    state: { layoutCursor: cursor, columnIndex: outCi, y: outY },
  }
}

export class BookSpread {
  spreadW = 1280
  spreadH = 720
  strokes: Stroke[] = []
  currentStroke: Stroke | null = null
  tool: ToolId = 'pen'
  color = '#2a4a7c'
  showText = true
  figureImage: HTMLImageElement | null = null

  preparedBefore: PreparedTextWithSegments | null = null
  preparedAfter: PreparedTextWithSegments | null = null
  preparedCaption: PreparedTextWithSegments | null = null
  preparedFigureLabel: PreparedTextWithSegments | null = null

  private mask: ImageData | null = null
  private maskDirty = true
  private strokeLayer: HTMLCanvasElement | null = null

  private ensureStrokeLayer(): HTMLCanvasElement {
    if (!this.strokeLayer || this.strokeLayer.width !== this.spreadW || this.strokeLayer.height !== this.spreadH) {
      const c = document.createElement('canvas')
      c.width = this.spreadW
      c.height = this.spreadH
      this.strokeLayer = c
    }
    return this.strokeLayer
  }

  constructor() {
    this.resetDefaultContent()
  }

  /** Prepare Norman excerpt + Carelman figure caption (constructor). */
  resetDefaultContent(): void {
    const norm = (s: string) => s.replace(/\s+/g, ' ').trim()
    this.preparedBefore = prepareWithSegments(norm(BODY_TEXT_BEFORE), BODY_FONT, { whiteSpace: 'normal' })
    this.preparedAfter = prepareWithSegments(norm(BODY_TEXT_AFTER), BODY_FONT, { whiteSpace: 'normal' })
    this.preparedCaption = prepareWithSegments(norm(FIGURE_CAPTION_BODY), CAPTION_FONT, { whiteSpace: 'normal' })
    this.preparedFigureLabel = prepareWithSegments(norm(FIGURE_LABEL), FIGURE_LABEL_FONT, { whiteSpace: 'normal' })
  }

  markStrokeDirty(): void {
    this.maskDirty = true
  }

  private ensureMask(): ImageData | null {
    if (this.currentStroke == null && !this.maskDirty && this.mask) {
      return this.mask
    }
    const list = this.currentStroke ? [...this.strokes, this.currentStroke] : this.strokes
    this.mask = rebuildMask(list, this.spreadW, this.spreadH)
    if (this.currentStroke == null) {
      this.maskDirty = false
    }
    return this.mask
  }

  private figureInlineGeometry(col: Column): {
    imgMaxW: number
    captionX: number
    captionW: number
    maxImgH: number
    imgDrawW: number
    imgDrawH: number
  } {
    const imgFrac = 0.44
    const gap = 8
    const imgMaxW = Math.floor(col.w * imgFrac)
    const captionX = col.x + imgMaxW + gap
    const captionW = Math.max(40, col.w - imgMaxW - gap)
    const maxImgH = Math.min(152, Math.floor(col.h * 0.5))
    let imgDrawW = imgMaxW
    let imgDrawH = 52
    if (this.figureImage?.complete && this.figureImage.naturalWidth) {
      const iw = this.figureImage.naturalWidth
      const ih = this.figureImage.naturalHeight
      const s = Math.min(imgMaxW / iw, maxImgH / ih)
      imgDrawW = iw * s
      imgDrawH = ih * s
    }
    return { imgMaxW, captionX, captionW, maxImgH, imgDrawW, imgDrawH }
  }

  /**
   * Dry-run caption (wrap beside image, full width below image, spill to following columns). Returns
   * skip bands so body text avoids the same vertical spans in each column.
   */
  private computeFigureCaptionBands(columns: Column[], mask: ImageData): ColumnBand[] {
    const col1 = columns[FIGURE_COLUMN_INDEX]
    const geo = this.figureInlineGeometry(col1)
    const imgY = col1.y + FIGURE_TOP_INSET
    const imgBottom = imgY + geo.imgDrawH
    const capTop = imgY + 6
    const slotTop = Math.max(col1.y, imgY - 4)
    const capBottomMin = Math.min(col1.y + col1.h - 4, mask.height - 4)

    if (capBottomMin <= capTop + CAPTION_LINE_HEIGHT) {
      return [
        {
          columnIndex: FIGURE_COLUMN_INDEX,
          top: slotTop,
          bottom: Math.min(col1.y + col1.h, imgBottom + FIGURE_AFTER_PAD),
        },
      ]
    }

    const { maxYPerColumn } = layoutFigureCaptionPretext(
      null,
      mask,
      this.preparedFigureLabel,
      this.preparedCaption,
      columns,
      FIGURE_COLUMN_INDEX,
      geo.captionX,
      geo.captionW,
      imgBottom,
      capTop,
    )

    return buildCaptionSkipBands(
      columns,
      FIGURE_COLUMN_INDEX,
      slotTop,
      imgBottom,
      capTop,
      maxYPerColumn,
    )
  }

  /** Photo under body text so column ink can cover the frame edges; caption is drawn later. */
  private drawFigurePhotoInColumn(ctx: CanvasRenderingContext2D, col: Column): void {
    const { imgMaxW, imgDrawW, imgDrawH } = this.figureInlineGeometry(col)
    const imgX = col.x + (imgMaxW - imgDrawW) / 2
    const imgY = col.y + FIGURE_TOP_INSET

    ctx.save()
    const px0 = Math.max(0, col.x)
    const py0 = Math.max(0, col.y)
    const px1 = Math.min(this.spreadW, col.x + col.w)
    const py1 = Math.min(this.spreadH, col.y + col.h)
    if (px1 > px0 && py1 > py0) {
      ctx.beginPath()
      ctx.rect(px0, py0, px1 - px0, py1 - py0)
      ctx.clip()
    }
    if (this.figureImage?.complete && this.figureImage.naturalWidth) {
      ctx.fillStyle = '#e8e4dc'
      ctx.fillRect(imgX - 3, imgY - 3, imgDrawW + 6, imgDrawH + 6)
      ctx.strokeStyle = 'rgba(55, 45, 35, 0.2)'
      ctx.lineWidth = 1
      ctx.strokeRect(imgX - 3, imgY - 3, imgDrawW + 6, imgDrawH + 6)
      ctx.drawImage(this.figureImage, imgX, imgY, imgDrawW, imgDrawH)
    } else {
      ctx.fillStyle = 'rgba(0,0,0,0.07)'
      ctx.fillRect(col.x, imgY, imgMaxW, 48)
    }
    ctx.restore()
  }

  /** Draw caption after body so it sits on top; may span the figure column and spill to the right. */
  private drawFigureCaption(ctx: CanvasRenderingContext2D, columns: Column[], mask: ImageData): void {
    const col1 = columns[FIGURE_COLUMN_INDEX]
    const geo = this.figureInlineGeometry(col1)
    const imgY = col1.y + FIGURE_TOP_INSET
    const imgBottom = imgY + geo.imgDrawH
    const capTop = imgY + 6
    ctx.save()
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.fillStyle = '#33302c'
    layoutFigureCaptionPretext(
      ctx,
      mask,
      this.preparedFigureLabel,
      this.preparedCaption,
      columns,
      FIGURE_COLUMN_INDEX,
      geo.captionX,
      geo.captionW,
      imgBottom,
      capTop,
    )
    ctx.restore()
  }

  private flowSpreadText(ctx: CanvasRenderingContext2D, mask: ImageData): void {
    const pb = this.preparedBefore
    const pa = this.preparedAfter
    if (!pb || !pa) return

    const textBottom = this.spreadH - TEXT_BOTTOM_PAD
    const textBandH = Math.max(LINE_HEIGHT * 3, textBottom - TEXT_TOP)

    ctx.save()
    ctx.beginPath()
    ctx.rect(0, TEXT_TOP, this.spreadW, textBandH)
    ctx.clip()

    const cols = buildColumnsBand(this.spreadW, TEXT_TOP, textBandH)
    const col1 = cols[FIGURE_COLUMN_INDEX]
    const captionBands = this.computeFigureCaptionBands(cols, mask)

    this.drawFigurePhotoInColumn(ctx, col1)

    const r1 = flowTextResumable(
      ctx,
      mask,
      pb,
      cols,
      { segmentIndex: 0, graphemeIndex: 0 },
      null,
      captionBands,
    )
    flowTextResumable(
      ctx,
      mask,
      pa,
      cols,
      { segmentIndex: 0, graphemeIndex: 0 },
      { columnIndex: r1.state.columnIndex, y: r1.state.y },
      captionBands,
    )

    this.drawFigureCaption(ctx, cols, mask)

    ctx.restore()
  }

  drawDecorations(ctx: CanvasRenderingContext2D): void {
    const { spreadW } = this
    ctx.save()
    ctx.fillStyle = '#6a5a4a'
    ctx.font = HEADING_FONT_SMALL
    ctx.textBaseline = 'alphabetic'
    const margin = TEXT_MARGIN_X
    ctx.fillText(CHAPTER_LABEL.toUpperCase(), margin, 36)
    ctx.textAlign = 'right'
    ctx.fillText(BOOK_TITLE.toUpperCase(), spreadW - margin, 36)
    ctx.textAlign = 'left'
    ctx.fillStyle = '#1c1c1c'
    ctx.font = TITLE_FONT
    const gutter = 40
    const innerW = spreadW - margin * 2
    const pageW = (innerW - gutter) / 2
    ctx.fillText(SECTION_HEADING, margin, 64)
    ctx.strokeStyle = 'rgba(90, 70, 50, 0.25)'
    ctx.lineWidth = 1
    const foldX = margin + pageW + gutter / 2
    ctx.beginPath()
    ctx.moveTo(foldX, 52)
    ctx.lineTo(foldX, this.spreadH - 40)
    ctx.stroke()
    ctx.restore()
  }

  /** Ink-like calligraphy: crisp edges, variable width, tapered ends — no blur / spray. */
  private paintBrushStroke(sc: CanvasRenderingContext2D, s: Stroke): void {
    const pts = s.points
    sc.save()
    sc.lineCap = 'round'
    sc.lineJoin = 'round'
    sc.globalCompositeOperation = 'source-over'
    sc.strokeStyle = s.color
    sc.shadowBlur = 0
    sc.globalAlpha = 1

    if (pts.length === 1) {
      const w = pts[0].lw ?? s.width
      const r = w / 2
      sc.fillStyle = s.color
      sc.beginPath()
      sc.arc(pts[0].x, pts[0].y, r, 0, Math.PI * 2)
      sc.fill()
      sc.restore()
      return
    }

    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i]
      const p1 = pts[i + 1]
      sc.lineWidth = brushSegmentCoreWidth(s, i)
      sc.beginPath()
      sc.moveTo(p0.x, p0.y)
      sc.lineTo(p1.x, p1.y)
      sc.stroke()
    }

    sc.restore()
  }

  private paintStrokePath(sc: CanvasRenderingContext2D, s: Stroke): void {
    if (s.points.length === 0) return
    if (s.tool === 'brush') {
      this.paintBrushStroke(sc, s)
      return
    }
    sc.lineCap = 'round'
    sc.lineJoin = 'round'
    if (s.points.length === 1) {
      const p = s.points[0]
      const r = s.width / 2
      if (s.tool === 'eraser') {
        sc.globalCompositeOperation = 'destination-out'
        sc.fillStyle = 'rgba(0,0,0,1)'
        sc.globalAlpha = 1
        sc.shadowBlur = 0
      } else {
        sc.globalCompositeOperation = 'source-over'
        sc.fillStyle = s.color
        sc.globalAlpha = s.tool === 'marker' ? 0.42 : 1
        sc.shadowBlur = 0
        sc.shadowColor = s.color
      }
      sc.beginPath()
      sc.arc(p.x, p.y, r, 0, Math.PI * 2)
      sc.fill()
      return
    }
    if (s.tool === 'eraser') {
      sc.globalCompositeOperation = 'destination-out'
      sc.strokeStyle = 'rgba(0,0,0,1)'
      sc.globalAlpha = 1
      sc.shadowBlur = 0
    } else {
      sc.globalCompositeOperation = 'source-over'
      sc.strokeStyle = s.color
      sc.globalAlpha = s.tool === 'marker' ? 0.42 : 1
      sc.shadowBlur = 0
    }
    sc.lineWidth = s.width
    sc.beginPath()
    sc.moveTo(s.points[0].x, s.points[0].y)
    for (let i = 1; i < s.points.length; i++) {
      sc.lineTo(s.points[i].x, s.points[i].y)
    }
    sc.stroke()
  }

  drawStrokes(ctx: CanvasRenderingContext2D): void {
    const layer = this.ensureStrokeLayer()
    const sc = layer.getContext('2d')!
    sc.clearRect(0, 0, this.spreadW, this.spreadH)
    for (const s of this.strokes) {
      this.paintStrokePath(sc, s)
    }
    if (this.currentStroke) {
      this.paintStrokePath(sc, this.currentStroke)
    }
    ctx.drawImage(layer, 0, 0)
  }

  render(ctx: CanvasRenderingContext2D): void {
    const mask = this.ensureMask()

    ctx.save()
    ctx.fillStyle = '#f2ebe0'
    ctx.fillRect(0, 0, this.spreadW, this.spreadH)
    ctx.fillStyle = 'rgba(255,255,255,0.35)'
    ctx.fillRect(0, 0, this.spreadW, 8)
    ctx.restore()

    this.drawDecorations(ctx)

    if (mask && this.showText && this.preparedBefore && this.preparedAfter) {
      this.flowSpreadText(ctx, mask)
    }

    this.drawStrokes(ctx)

    ctx.save()
    ctx.strokeStyle = 'rgba(80, 60, 45, 0.12)'
    ctx.strokeRect(0.5, 0.5, this.spreadW - 1, this.spreadH - 1)
    ctx.restore()
  }

  beginStroke(x: number, y: number): void {
    const w =
      this.tool === 'marker' ? 14 : this.tool === 'brush' ? 7.5 : this.tool === 'eraser' ? 24 : 3.2
    this.currentStroke = {
      tool: this.tool,
      color: this.color,
      width: w,
      points: this.tool === 'brush' ? [{ x, y, lw: w }] : [{ x, y }],
    }
  }

  extendStroke(x: number, y: number): void {
    const s = this.currentStroke
    if (!s) return
    const last = s.points[s.points.length - 1]
    const dx = x - last.x
    const dy = y - last.y
    if (dx * dx + dy * dy < 1.5) return
    if (s.tool === 'brush') {
      const dist = Math.hypot(dx, dy)
      const t = clamp((dist - 2) / 34, 0, 1)
      const lw = s.width * (1.2 - 0.4 * t)
      s.points.push({ x, y, lw })
    } else {
      s.points.push({ x, y })
    }
  }

  endStroke(): void {
    if (!this.currentStroke || this.currentStroke.points.length < 1) {
      this.currentStroke = null
      return
    }
    this.strokes.push(this.currentStroke)
    this.currentStroke = null
    this.markStrokeDirty()
  }

  clearDoodles(): void {
    this.strokes = []
    this.currentStroke = null
    this.markStrokeDirty()
  }

  undoStroke(): void {
    this.strokes.pop()
    this.markStrokeDirty()
  }

  toolLineWidth(): number {
    switch (this.tool) {
      case 'marker':
        return 14
      case 'brush':
        return 7.5
      case 'eraser':
        return 24
      default:
        return 3.2
    }
  }
}
