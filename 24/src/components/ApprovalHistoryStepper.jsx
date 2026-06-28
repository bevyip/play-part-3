import { Fragment } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { APPROVAL_HISTORY } from "../data/approvalHistory";
import { TABLE_TEXT_COLOR } from "../theme/colors";

const STEPPER_CIRCLE_SIZE_PX = 12;
const STEPPER_CONNECTOR_GAP_PX = 10;
const STEPPER_ENTRY_SPACING_PX = 56;
const STEPPER_WRAPPED_ROW_GAP_PX = 14;
const STEPPER_WRAPPED_MIN_ENTRY_HEIGHT_PX = 56;
const STEPPER_CONNECTOR_WIDTH_PX = 2;
const STEPPER_MIDDLE_COL_PX = 18;
const STEPPER_COLUMN_GAP_PX = 12;
const STEPPER_WRAPPED_COLUMN_GAP_PX = 6;
const STEPPER_WRAPPED_LEFT_COL_MAX_PX = 108;
const STEPPER_WRAPPED_RIGHT_COL_MAX_PX = 72;

const stepperTextBaseSx = {
  fontSize: 14,
  fontWeight: 400,
  m: 0,
  color: TABLE_TEXT_COLOR,
};

const stepperNowrapTextSx = {
  ...stepperTextBaseSx,
  lineHeight: `${STEPPER_CIRCLE_SIZE_PX}px`,
  whiteSpace: "nowrap",
};

const stepperWrappedLeftTextSx = {
  ...stepperTextBaseSx,
  fontSize: 13,
  lineHeight: 1.35,
  whiteSpace: "normal",
  wordBreak: "break-word",
  overflowWrap: "anywhere",
  minWidth: 0,
};

const stepperWrappedRightTextSx = {
  ...stepperTextBaseSx,
  fontSize: 13,
  lineHeight: 1.35,
  whiteSpace: "normal",
  wordBreak: "break-word",
  overflowWrap: "anywhere",
  minWidth: 0,
};

function HistoryLeftLabel({ entry, sx, spacingBottom = 0 }) {
  if (!entry.date) {
    return (
      <Typography
        sx={{ ...sx, pb: spacingBottom ? `${spacingBottom}px` : 0 }}
        component="div"
      >
        {entry.action}
      </Typography>
    );
  }

  return (
    <Box
      sx={{
        minWidth: 0,
        textAlign: "right",
        pb: spacingBottom ? `${spacingBottom}px` : 0,
      }}
    >
      <Typography sx={sx} component="div">
        {entry.action}
      </Typography>
      <Typography sx={sx} component="div">
        {entry.date}
      </Typography>
    </Box>
  );
}

function ApprovalStepIcon({ color }) {
  return (
    <Box
      sx={{
        width: STEPPER_CIRCLE_SIZE_PX,
        height: STEPPER_CIRCLE_SIZE_PX,
        borderRadius: "50%",
        border: `2px solid ${color}`,
        bgcolor: "#fff",
        boxSizing: "border-box",
        flexShrink: 0,
        position: "relative",
        zIndex: 1,
      }}
    />
  );
}

function formatHistoryLabel(entry) {
  if (entry.date) {
    return `${entry.action} ${entry.date}`;
  }
  return entry.action;
}

export default function ApprovalHistoryStepper({
  history = APPROVAL_HISTORY,
  maxWidth = 300,
  fullWidth = false,
  preventTextWrap = false,
}) {
  const leftTextSx = preventTextWrap
    ? stepperNowrapTextSx
    : stepperWrappedLeftTextSx;
  const rightTextSx = preventTextWrap
    ? stepperNowrapTextSx
    : stepperWrappedRightTextSx;

  return (
    <Box
      sx={{
        width: preventTextWrap ? (fullWidth ? "100%" : "max-content") : "100%",
        maxWidth,
        minWidth: 0,
        mx: preventTextWrap ? "auto" : 0,
        py: 1,
        pb: 1.5,
        boxSizing: "border-box",
        overflow: preventTextWrap ? "visible" : "hidden",
      }}
    >
      <Box
        sx={{
          display: "grid",
          width: preventTextWrap ? "100%" : "fit-content",
          maxWidth: "100%",
          minWidth: 0,
          gridTemplateColumns: preventTextWrap
            ? `auto ${STEPPER_MIDDLE_COL_PX}px auto`
            : `minmax(0, ${STEPPER_WRAPPED_LEFT_COL_MAX_PX}px) ${STEPPER_MIDDLE_COL_PX}px minmax(0, ${STEPPER_WRAPPED_RIGHT_COL_MAX_PX}px)`,
          columnGap: preventTextWrap
            ? `${STEPPER_COLUMN_GAP_PX}px`
            : `${STEPPER_WRAPPED_COLUMN_GAP_PX}px`,
          alignItems: "start",
        }}
      >
        {history.map((h, i) => {
          const isLastEntry = i === history.length - 1;
          const wrappedRowSpacing =
            !preventTextWrap && !isLastEntry ? STEPPER_WRAPPED_ROW_GAP_PX : 0;

          return (
            <Fragment key={`${h.action}-${h.date}-${i}`}>
              {preventTextWrap ? (
                <Typography
                  sx={{
                    ...leftTextSx,
                    textAlign: "right",
                  }}
                >
                  {formatHistoryLabel(h)}
                </Typography>
              ) : (
                <HistoryLeftLabel
                  entry={h}
                  sx={leftTextSx}
                  spacingBottom={wrappedRowSpacing}
                />
              )}

              <Box
                sx={{
                  position: "relative",
                  display: "flex",
                  justifyContent: "center",
                  alignSelf: "stretch",
                  minHeight: isLastEntry
                    ? STEPPER_CIRCLE_SIZE_PX
                    : preventTextWrap
                      ? STEPPER_ENTRY_SPACING_PX
                      : STEPPER_WRAPPED_MIN_ENTRY_HEIGHT_PX,
                }}
              >
                <ApprovalStepIcon color={h.color} />
                {i < history.length - 1 ? (
                  <Box
                    sx={{
                      position: "absolute",
                      top: STEPPER_CIRCLE_SIZE_PX + STEPPER_CONNECTOR_GAP_PX,
                      bottom: STEPPER_CONNECTOR_GAP_PX,
                      left: "50%",
                      transform: "translateX(-50%)",
                      width: STEPPER_CONNECTOR_WIDTH_PX,
                      bgcolor: "#d1d5db",
                      pointerEvents: "none",
                    }}
                  />
                ) : null}
              </Box>

              <Typography
                sx={{
                  ...rightTextSx,
                  pl: preventTextWrap ? 0.5 : 0,
                  pb: wrappedRowSpacing ? `${wrappedRowSpacing}px` : 0,
                }}
              >
                {h.person}
              </Typography>
            </Fragment>
          );
        })}
      </Box>
    </Box>
  );
}
