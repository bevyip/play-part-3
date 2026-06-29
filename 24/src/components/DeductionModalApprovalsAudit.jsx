import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import {
  KeyboardArrowDownIcon,
  KeyboardArrowUpIcon,
  buttonChevronSx,
} from "../theme/icons";
import ModalAuditLogPanel from "./ModalAuditLogPanel";
import ApprovalManageMenu from "./ApprovalManageMenu";
import ApprovalActionModal from "./ApprovalActionModal";
import { TABLE_TEXT_COLOR } from "../theme/colors";
import {
  getAssignmentDisplay,
  isAwaitingCurrentUserApproval,
} from "../utils/approvalHelpers";
import {
  NEW_DESIGN_APPROVE_GREEN,
  NEW_DESIGN_REJECT_RED,
  newDesignOutlinedButtonHoverSx,
  newDesignPrimaryOutlinedButtonHoverSx,
} from "../theme/newDesignActionButtons";

const LABEL_COLOR = "#6b7280";
const GREEN = NEW_DESIGN_APPROVE_GREEN;
const REJECT_RED = NEW_DESIGN_REJECT_RED;
const DISABLED_TEXT = "#9ca3af";
const DISABLED_BORDER = "#e0e0e0";

const sectionCapsLabelSx = {
  fontSize: 11,
  fontWeight: 600,
  color: LABEL_COLOR,
  letterSpacing: "0.03em",
  textTransform: "uppercase",
  lineHeight: 1.2,
};

const subsectionTitleSx = {
  fontSize: 15,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  mb: 1.5,
};

const innerPanelSx = {
  border: "1px solid #e8eaed",
  borderRadius: 1,
  bgcolor: "#fff",
  px: 2,
  py: 2,
};

const flatOutlinedSx = (borderColor, color) => ({
  boxShadow: "none",
  fontWeight: 400,
  "&:hover": { borderColor, color, bgcolor: "transparent", boxShadow: "none" },
  "&:active": { borderColor, color, bgcolor: "transparent", boxShadow: "none" },
});

const actionButtonBaseSx = {
  fontSize: 14,
  fontWeight: 400,
  px: 1.25,
  minHeight: 32,
  py: 0.5,
  lineHeight: 1.2,
  borderRadius: 1,
};

const enabledApproveSx = {
  ...actionButtonBaseSx,
  color: GREEN,
  borderColor: GREEN,
  ...newDesignOutlinedButtonHoverSx(GREEN),
};

const enabledRejectSx = {
  ...actionButtonBaseSx,
  color: REJECT_RED,
  borderColor: REJECT_RED,
  ...newDesignOutlinedButtonHoverSx(REJECT_RED),
};

const enabledMoreSx = {
  ...actionButtonBaseSx,
  color: "primary.main",
  borderColor: "primary.main",
  ...newDesignPrimaryOutlinedButtonHoverSx,
};

const disabledActionSx = {
  ...actionButtonBaseSx,
  color: DISABLED_TEXT,
  borderColor: DISABLED_BORDER,
  ...flatOutlinedSx(DISABLED_BORDER, DISABLED_TEXT),
  pointerEvents: "none",
};

export default function DeductionModalApprovalsAudit({
  approvalStatus,
  approvalHistory,
  onApprove,
  onReject,
  onReassign,
  onUnassign,
}) {
  const [moreAnchorEl, setMoreAnchorEl] = useState(null);
  const [actionModalType, setActionModalType] = useState(null);

  const canApproveReject = isAwaitingCurrentUserApproval(approvalStatus);
  const assignmentDisplay = getAssignmentDisplay(approvalStatus);

  useEffect(() => {
    setActionModalType(null);
  }, [approvalStatus]);

  const handleMoreOpen = (event) => {
    setMoreAnchorEl(event.currentTarget);
  };

  const handleMoreClose = () => {
    setMoreAnchorEl(null);
  };

  const openActionModal = (type) => {
    setActionModalType(type);
  };

  const closeActionModal = () => {
    setActionModalType(null);
  };

  const handleActionConfirm = (payload) => {
    if (actionModalType === "reject") {
      onReject?.(payload);
    } else if (actionModalType === "reassign") {
      onReassign?.(payload);
    } else if (actionModalType === "unassign") {
      onUnassign?.(payload);
    }
  };

  return (
    <Box
      sx={{
        bgcolor: "#fff",
        borderRadius: 1,
        border: "1px solid #e8eaed",
        boxShadow: "0 1px 4px rgba(0, 0, 0, 0.06)",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
        px: 2.5,
        py: 2,
      }}
    >
      <Typography
        sx={{
          fontSize: 16,
          fontWeight: 400,
          color: TABLE_TEXT_COLOR,
          mb: 2,
        }}
      >
        Approvals &amp; Audit Log
      </Typography>

      <Typography sx={{ ...subsectionTitleSx, mb: 1.25 }}>Approvals</Typography>
      <Box sx={{ ...innerPanelSx, mb: 2.5 }}>
        <Typography sx={{ ...sectionCapsLabelSx, mb: 0.75 }}>
          Assignments
        </Typography>

        <Typography
          sx={{
            fontSize: 14,
            fontWeight: 400,
            color: TABLE_TEXT_COLOR,
            mb: 2.5,
            lineHeight: 1.45,
          }}
        >
          <Box component="span" sx={{ fontWeight: 600 }}>
            Assigned:
          </Box>{" "}
          {assignmentDisplay}
        </Typography>

        <Typography sx={{ ...sectionCapsLabelSx, mb: 1 }}>Actions</Typography>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            flexWrap: "wrap",
          }}
        >
          <Button
            disableRipple
            variant="outlined"
            size="small"
            onClick={canApproveReject ? onApprove : undefined}
            sx={canApproveReject ? enabledApproveSx : disabledActionSx}
          >
            APPROVE
          </Button>
          <Button
            disableRipple
            variant="outlined"
            size="small"
            onClick={
              canApproveReject ? () => openActionModal("reject") : undefined
            }
            sx={canApproveReject ? enabledRejectSx : disabledActionSx}
          >
            REJECT
          </Button>
          <Button
            disableRipple
            variant="outlined"
            size="small"
            endIcon={
              moreAnchorEl ? (
                <KeyboardArrowUpIcon
                  sx={{ ...buttonChevronSx, color: "inherit" }}
                />
              ) : (
                <KeyboardArrowDownIcon
                  sx={{ ...buttonChevronSx, color: "inherit" }}
                />
              )
            }
            onClick={handleMoreOpen}
            sx={enabledMoreSx}
          >
            MORE
          </Button>
        </Box>

        <ApprovalManageMenu
          anchorEl={moreAnchorEl}
          open={Boolean(moreAnchorEl)}
          onClose={handleMoreClose}
          approvalStatus={approvalStatus}
          variant="more"
          onReassignClick={() => openActionModal("reassign")}
          onUnassignClick={() => openActionModal("unassign")}
          anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
          transformOrigin={{ vertical: "top", horizontal: "center" }}
        />
      </Box>

      <Typography sx={{ ...subsectionTitleSx, mb: 1.25 }}>Audit Log</Typography>
      <ModalAuditLogPanel approvalHistory={approvalHistory} />

      <ApprovalActionModal
        open={Boolean(actionModalType)}
        actionType={actionModalType}
        onClose={closeActionModal}
        onConfirm={handleActionConfirm}
      />
    </Box>
  );
}
