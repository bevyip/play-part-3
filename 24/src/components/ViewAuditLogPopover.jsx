import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import CloseIcon from "@mui/icons-material/Close";
import ModalAuditLogPanel from "./ModalAuditLogPanel";
import { TABLE_TEXT_COLOR } from "../theme/colors";

export default function ViewAuditLogPopover({
  open,
  onClose,
  approvalHistory,
}) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth={false}
      PaperProps={{
        sx: {
          borderRadius: 1,
          width: 680,
          maxWidth: "calc(100vw - 32px)",
          maxHeight: "min(720px, calc(100vh - 80px))",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          m: 2,
        },
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 2.5,
          pt: 2,
          pb: 0,
          flexShrink: 0,
        }}
      >
        <Typography
          sx={{
            fontSize: 16,
            fontWeight: 400,
            color: TABLE_TEXT_COLOR,
          }}
        >
          Audit Log
        </Typography>
        <IconButton size="small" onClick={onClose} sx={{ p: 0.5 }}>
          <CloseIcon sx={{ fontSize: 20, color: "#6b7280" }} />
        </IconButton>
      </Box>

      <DialogContent
        sx={{
          p: 0,
          px: 2.5,
          pt: 2,
          pb: 2,
          overflowY: "auto",
          flex: 1,
        }}
      >
        <ModalAuditLogPanel
          approvalHistory={approvalHistory}
          timelineMaxHeight={440}
        />
      </DialogContent>
    </Dialog>
  );
}
