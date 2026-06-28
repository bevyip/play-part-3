import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Collapse from "@mui/material/Collapse";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import ForumIcon from "@mui/icons-material/Forum";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import { ArrowDropDownIcon, buttonChevronSx } from "../theme/icons";
import { TABLE_TEXT_COLOR } from "../theme/colors";
import { simpleTooltipProps } from "../theme/tooltips";
import ApprovalManageMenu from "./ApprovalManageMenu";
import ApprovalActionModal from "./ApprovalActionModal";
import {
  getAssignmentDisplay,
  getApprovalStatusDisplayWithColor,
  isAwaitingCurrentUserApproval,
} from "../utils/approvalHelpers";

const LABEL_COLOR = "#6b7280";
const ICON_GREY = "#6b7280";

const rowLabelSx = {
  fontSize: 14,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  lineHeight: 1.45,
};

const manageButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  px: 1.5,
  minWidth: 90,
  minHeight: 32,
  py: 0.25,
  lineHeight: 1.2,
  borderRadius: 1,
  color: "#fff",
  boxShadow: "none",
  "&:hover": { boxShadow: "none", bgcolor: "primary.dark", color: "#fff" },
  "& .MuiButton-endIcon": { color: "#fff" },
};

export default function MiniApprovalComponent({
  approvalStatus,
  onApprove,
  onReject,
  onReassign,
  onUnassign,
  onCommentsClick,
}) {
  const assignedToMe = isAwaitingCurrentUserApproval(approvalStatus);
  const [expanded, setExpanded] = useState(assignedToMe);
  const [manageAnchorEl, setManageAnchorEl] = useState(null);
  const [actionModalType, setActionModalType] = useState(null);

  const assignmentDisplay = getAssignmentDisplay(approvalStatus);
  const approvalStatusInfo = getApprovalStatusDisplayWithColor(approvalStatus);

  useEffect(() => {
    setActionModalType(null);
  }, [approvalStatus]);

  const handleManageOpen = (event) => {
    setManageAnchorEl(event.currentTarget);
  };

  const handleManageClose = () => {
    setManageAnchorEl(null);
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
        border: "1px solid #e8eaed",
        borderRadius: 1,
        bgcolor: "#fff",
        px: 1.75,
        py: 2,
        mb: 2,
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 1,
          minHeight: 32,
        }}
      >
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography sx={rowLabelSx}>
            <Box component="span" sx={{ fontWeight: 600 }}>
              Assigned:
            </Box>{" "}
            {assignmentDisplay}
          </Typography>
        </Box>

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.75,
            flexShrink: 0,
          }}
        >
          <Button
            disableRipple
            variant="outlined"
            size="small"
            sx={{
              fontSize: 13,
              fontWeight: 400,
              color: "primary.main",
              borderColor: "primary.main",
              minHeight: 32,
              py: 0,
              px: 0,
              boxShadow: "none",
              overflow: "hidden",
              "&:hover": {
                bgcolor: "transparent",
                borderColor: "primary.main",
                boxShadow: "none",
              },
            }}
          >
            <Box
              component="span"
              sx={{
                px: 1.25,
                display: "flex",
                alignItems: "center",
                alignSelf: "stretch",
                py: 0.5,
              }}
            >
              COMPLETE
            </Box>
            <Box
              component="span"
              sx={{
                display: "flex",
                alignItems: "center",
                alignSelf: "stretch",
                borderLeft: "1px solid",
                borderColor: "primary.main",
                px: 0.5,
                py: 0.5,
              }}
            >
              <ArrowDropDownIcon sx={buttonChevronSx} />
            </Box>
          </Button>
          <IconButton
            disableRipple
            size="small"
            onClick={() => setExpanded((prev) => !prev)}
            sx={{ p: 0.25, color: LABEL_COLOR }}
          >
            {expanded ? (
              <KeyboardArrowUpIcon sx={{ fontSize: 22, color: LABEL_COLOR }} />
            ) : (
              <KeyboardArrowDownIcon
                sx={{ fontSize: 22, color: LABEL_COLOR }}
              />
            )}
          </IconButton>
        </Box>
      </Box>

      <Collapse
        in={expanded}
        timeout={{ enter: 280, exit: 220 }}
        easing={{
          enter: "cubic-bezier(0.4, 0, 0.2, 1)",
          exit: "cubic-bezier(0.4, 0, 0.2, 1)",
        }}
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 1,
            pt: 1.25,
            minHeight: 32,
          }}
        >
          <Typography sx={rowLabelSx}>
            <Box component="span" sx={{ fontWeight: 600 }}>
              Approval Status:
            </Box>{" "}
            <Box component="span" sx={{ color: approvalStatusInfo.color }}>
              {approvalStatusInfo.text}
            </Box>
          </Typography>

          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.75,
              flexShrink: 0,
            }}
          >
            <Tooltip title="View Audit Log" {...simpleTooltipProps}>
              <IconButton
                disableRipple
                size="small"
                onClick={onCommentsClick}
                sx={{ p: 0.25, color: ICON_GREY }}
              >
                <ForumIcon sx={{ fontSize: 20, color: ICON_GREY }} />
              </IconButton>
            </Tooltip>
            <Button
              disableRipple
              variant="contained"
              size="small"
              endIcon={
                <ArrowDropDownIcon sx={{ ...buttonChevronSx, color: "#fff" }} />
              }
              onClick={handleManageOpen}
              sx={manageButtonSx}
            >
              MANAGE
            </Button>
          </Box>
        </Box>
      </Collapse>

      <ApprovalManageMenu
        anchorEl={manageAnchorEl}
        open={Boolean(manageAnchorEl)}
        onClose={handleManageClose}
        approvalStatus={approvalStatus}
        variant="manage"
        onApprove={onApprove}
        onReject={() => openActionModal("reject")}
        onReassignClick={() => openActionModal("reassign")}
        onUnassignClick={() => openActionModal("unassign")}
      />

      <ApprovalActionModal
        open={Boolean(actionModalType)}
        actionType={actionModalType}
        onClose={closeActionModal}
        onConfirm={handleActionConfirm}
      />
    </Box>
  );
}
