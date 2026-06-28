import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import ModalAuditLogTimeline from "./ModalAuditLogTimeline";
import ModalCommentsSection from "./ModalCommentsSection";

const LABEL_COLOR = "#6b7280";

const sectionCapsLabelSx = {
  fontSize: 11,
  fontWeight: 600,
  color: LABEL_COLOR,
  letterSpacing: "0.03em",
  textTransform: "uppercase",
  lineHeight: 1.2,
};

const auditLogPanelSx = {
  border: "1px solid #e8eaed",
  borderRadius: 1,
  bgcolor: "#fff",
  px: 2,
  py: 2,
};

export default function ModalAuditLogPanel({
  approvalHistory = [],
  timelineMaxHeight,
}) {
  return (
    <Box sx={auditLogPanelSx}>
      <Typography sx={{ ...sectionCapsLabelSx, mb: 1.5 }}>
        Approval History
      </Typography>
      <ModalAuditLogTimeline
        history={approvalHistory}
        maxHeight={timelineMaxHeight}
      />

      <Typography sx={{ ...sectionCapsLabelSx, mt: 2.5, mb: 1.5 }}>
        Comments
      </Typography>
      <ModalCommentsSection variant="embedded" />
    </Box>
  );
}
