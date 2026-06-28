import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { TABLE_TEXT_COLOR } from "../theme/colors";
import {
  formatAuditLogTimestamp,
  formatAuditLogPrimaryText,
} from "../data/approvalHistory";

const LABEL_COLOR = "#6b7280";

// Stepper rail
const DOT_SIZE = 12;
const LINE_WIDTH = 2;
const CONNECTOR_GAP = 10;
const DOT_COLUMN_WIDTH = DOT_SIZE + 16;

// Text metrics
const PRIMARY_FONT_SIZE = 14;
const PRIMARY_LINE_HEIGHT = 1.45;
const PRIMARY_ROW_HEIGHT = PRIMARY_FONT_SIZE * PRIMARY_LINE_HEIGHT;

// Fixed entry heights
const ENTRY_BASE_HEIGHT = 48;
const COMMENT_EXTRA_HEIGHT = 24;
const COMMENT_TOP = PRIMARY_ROW_HEIGHT + 6;

const primaryTextSx = {
  fontSize: PRIMARY_FONT_SIZE,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  lineHeight: PRIMARY_LINE_HEIGHT,
};

const commentTextSx = {
  fontSize: 14,
  fontStyle: "italic",
  color: LABEL_COLOR,
  lineHeight: 1.4,
};

const auditLogScrollSx = {
  maxHeight: 320,
  overflowY: "auto",
  overflowX: "hidden",
  px: 4.5,
  py: 3,
  border: "1px solid #e8eaed",
  borderRadius: 1,
  scrollbarWidth: "thin",
  scrollbarColor: "#bdbdbd #f0f0f0",
  "&::-webkit-scrollbar": { width: 8 },
  "&::-webkit-scrollbar-track": { backgroundColor: "#f0f0f0", borderRadius: 4 },
  "&::-webkit-scrollbar-thumb": { backgroundColor: "#bdbdbd", borderRadius: 4 },
  "&::-webkit-scrollbar-thumb:hover": { backgroundColor: "#9e9e9e" },
  "&::-webkit-scrollbar-button": { display: "none", width: 0, height: 0 },
};

function getEntryHeight(hasComment, isLast) {
  if (isLast) {
    return hasComment
      ? PRIMARY_ROW_HEIGHT + COMMENT_EXTRA_HEIGHT
      : PRIMARY_ROW_HEIGHT;
  }
  return hasComment
    ? ENTRY_BASE_HEIGHT + COMMENT_EXTRA_HEIGHT
    : ENTRY_BASE_HEIGHT;
}

function StepDot({ color }) {
  return (
    <Box
      sx={{
        width: DOT_SIZE,
        height: DOT_SIZE,
        borderRadius: "50%",
        border: `2px solid ${color || "#d1d5db"}`,
        bgcolor: "#fff",
        boxSizing: "border-box",
        flexShrink: 0,
        zIndex: 1,
      }}
    />
  );
}

function TimelineEntry({ entry, isLast }) {
  const hasComment = Boolean(entry.comment);
  const entryHeight = getEntryHeight(hasComment, isLast);
  const primaryText = formatAuditLogPrimaryText(entry);
  const timestamp = formatAuditLogTimestamp(entry.date);
  const [namePart, ...restParts] = primaryText.split(" · ");
  const actionPart = restParts.join(" · ");
  const unassignReason = entry.reason;
  const showUnassignReason =
    Boolean(unassignReason) &&
    entry.action?.replace(/ at$/i, "") === "Unassigned";

  const connectorTop = PRIMARY_ROW_HEIGHT / 2 + DOT_SIZE / 2 + CONNECTOR_GAP;

  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: `${DOT_COLUMN_WIDTH}px 1fr auto`,
        columnGap: 2,
        height: entryHeight,
      }}
    >
      <Box sx={{ position: "relative", height: entryHeight }}>
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: PRIMARY_ROW_HEIGHT,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <StepDot color={entry.color} />
        </Box>
        {!isLast && (
          <Box
            sx={{
              position: "absolute",
              top: connectorTop,
              bottom: CONNECTOR_GAP,
              left: "50%",
              transform: "translateX(-50%)",
              width: LINE_WIDTH,
              bgcolor: "#d1d5db",
              pointerEvents: "none",
            }}
          />
        )}
      </Box>

      <Box
        sx={{
          position: "relative",
          height: entryHeight,
          pr: 1.5,
          overflow: "hidden",
        }}
      >
        <Typography component="div" sx={primaryTextSx}>
          <Box component="span" sx={{ fontWeight: 600 }}>
            {namePart}
          </Box>
          {actionPart ? (
            <>
              {" · "}
              {showUnassignReason ? (
                <>
                  Unassigned{" "}
                  <Box component="span" sx={{ fontStyle: "italic" }}>
                    ({unassignReason})
                  </Box>
                </>
              ) : (
                actionPart
              )}
            </>
          ) : null}
        </Typography>
        {hasComment ? (
          <Typography
            component="div"
            sx={{ ...commentTextSx, position: "absolute", top: COMMENT_TOP }}
          >
            &ldquo;{entry.comment}&rdquo;
          </Typography>
        ) : null}
      </Box>

      <Box
        sx={{
          height: PRIMARY_ROW_HEIGHT,
          display: "flex",
          alignItems: "center",
          pl: 1.5,
        }}
      >
        <Typography
          sx={{
            fontSize: 12,
            fontWeight: 400,
            color: LABEL_COLOR,
            whiteSpace: "nowrap",
            lineHeight: PRIMARY_LINE_HEIGHT,
          }}
        >
          {timestamp}
        </Typography>
      </Box>
    </Box>
  );
}

export default function ModalAuditLogTimeline({
  history = [],
  maxHeight = 320,
}) {
  return (
    <Box sx={{ ...auditLogScrollSx, maxHeight }}>
      {history.map((entry, index) => (
        <TimelineEntry
          key={`${entry.action}-${entry.date}-${index}`}
          entry={entry}
          isLast={index === history.length - 1}
        />
      ))}
    </Box>
  );
}
