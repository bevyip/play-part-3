import './style.css'
import { jsPDF } from 'jspdf'
import { BookSpread, type ToolId } from './book'
import { FIGURE_IMAGE_PATH } from './content'

const canvas = document.querySelector<HTMLCanvasElement>('#spread')!
const ctx = canvas.getContext('2d')!
const book = new BookSpread()

const DPR_CAP = 2

function syncCanvasSize(): void {
  document.documentElement.style.setProperty('--spread-ar', String(book.spreadW / book.spreadH))
  const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP)
  const bw = book.spreadW
  const bh = book.spreadH
  canvas.width = Math.round(bw * dpr)
  canvas.height = Math.round(bh * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
}

function redraw(): void {
  book.render(ctx)
}

let raf = 0
function scheduleRedraw(): void {
  cancelAnimationFrame(raf)
  raf = requestAnimationFrame(() => {
    redraw()
  })
}

function canvasPoint(e: PointerEvent): { x: number; y: number } {
  const r = canvas.getBoundingClientRect()
  const sx = book.spreadW / r.width
  const sy = book.spreadH / r.height
  return {
    x: (e.clientX - r.left) * sx,
    y: (e.clientY - r.top) * sy,
  }
}

canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return
  canvas.setPointerCapture(e.pointerId)
  const p = canvasPoint(e)
  book.beginStroke(p.x, p.y)
  book.markStrokeDirty()
  scheduleRedraw()
})

canvas.addEventListener('pointermove', (e) => {
  if (!book.currentStroke) return
  const p = canvasPoint(e)
  book.extendStroke(p.x, p.y)
  book.markStrokeDirty()
  scheduleRedraw()
})

function endDraw(e: PointerEvent): void {
  if (!book.currentStroke) return
  canvas.releasePointerCapture(e.pointerId)
  book.endStroke()
  scheduleRedraw()
}

canvas.addEventListener('pointerup', endDraw)
canvas.addEventListener('pointercancel', endDraw)

document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-tool]').forEach((b) => b.classList.remove('active'))
    btn.classList.add('active')
    book.tool = btn.dataset.tool as ToolId
  })
})

document.querySelectorAll<HTMLButtonElement>('.swatch[data-color]').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.swatch[data-color]').forEach((b) => b.classList.remove('active'))
    btn.classList.add('active')
    book.color = btn.dataset.color ?? book.color
  })
})

const customColor = document.querySelector<HTMLInputElement>('#color-custom')!
customColor.addEventListener('input', () => {
  book.color = customColor.value
  document.querySelectorAll('.swatch[data-color]').forEach((b) => b.classList.remove('active'))
})

document.querySelector('#btn-clear-doodles')?.addEventListener('click', () => {
  book.clearDoodles()
  scheduleRedraw()
})

document.querySelector('#btn-undo')?.addEventListener('click', () => {
  book.undoStroke()
  scheduleRedraw()
})

const fig = new Image()
fig.decoding = 'async'
fig.src = FIGURE_IMAGE_PATH
fig.onload = () => {
  book.figureImage = fig
  scheduleRedraw()
}

document.querySelector('#btn-pdf-out')?.addEventListener('click', () => {
  const exportCanvas = document.createElement('canvas')
  exportCanvas.width = book.spreadW * 2
  exportCanvas.height = book.spreadH * 2
  const x = exportCanvas.getContext('2d')!
  x.scale(2, 2)
  book.render(x)
  const img = exportCanvas.toDataURL('image/png')
  const pdf = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' })
  const pageW = pdf.internal.pageSize.getWidth()
  const pageH = pdf.internal.pageSize.getHeight()
  pdf.addImage(img, 'PNG', 0, 0, pageW, pageH, undefined, 'FAST')
  pdf.save('art.pdf')
})

window.addEventListener('resize', () => {
  syncCanvasSize()
  scheduleRedraw()
})

const shell = document.querySelector('.spread-shell')
if (shell && typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(() => {
    syncCanvasSize()
    scheduleRedraw()
  }).observe(shell)
}

syncCanvasSize()
redraw()
