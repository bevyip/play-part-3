import Typography from "@mui/material/Typography";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Divider from "@mui/material/Divider";
import { TABLE_TEXT_COLOR } from "../theme/colors";
import { approvalMenuPaperSx } from "../theme/icons";
import { getManageMenuOptions } from "../utils/approvalHelpers";

const LABEL_COLOR = "#6b7280";
const DISABLED_TEXT = "#9ca3af";

const MENU_HORIZONTAL_PADDING = 1.75;

const menuTitleSx = {
  fontSize: 11,
  fontWeight: 600,
  color: LABEL_COLOR,
  letterSpacing: "0.03em",
  textTransform: "uppercase",
  textAlign: "left",
  px: MENU_HORIZONTAL_PADDING,
  pt: 0.75,
  pb: 0.5,
};

const menuItemSx = {
  fontSize: 13,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  letterSpacing: "0.02em",
  justifyContent: "flex-start",
  textAlign: "left",
  px: MENU_HORIZONTAL_PADDING,
  py: 0.5,
  minHeight: 28,
  minWidth: 0,
};

const disabledMenuItemSx = {
  ...menuItemSx,
  color: DISABLED_TEXT,
  opacity: 1,
  cursor: "default",
  "&.Mui-disabled": {
    opacity: 1,
    color: DISABLED_TEXT,
  },
};

export default function ApprovalManageMenu({
  anchorEl,
  open,
  onClose,
  approvalStatus,
  variant = "manage",
  onApprove,
  onReject,
  onReassignClick,
  onUnassignClick,
  anchorOrigin = { vertical: "bottom", horizontal: "right" },
  transformOrigin = { vertical: "top", horizontal: "right" },
}) {
  const options = getManageMenuOptions(approvalStatus);
  const showActions = variant === "manage";

  const handleApprove = () => {
    if (!options.approve) return;
    onClose();
    onApprove?.();
  };

  const handleReject = () => {
    if (!options.reject) return;
    onClose();
    onReject?.();
  };

  const handleReassign = () => {
    if (!options.reassign) return;
    onClose();
    onReassignClick?.();
  };

  const handleUnassign = () => {
    if (!options.unassign) return;
    onClose();
    onUnassignClick?.();
  };

  return (
    <Menu
      anchorEl={anchorEl}
      open={open}
      onClose={onClose}
      anchorOrigin={anchorOrigin}
      transformOrigin={transformOrigin}
      PaperProps={{
        sx: approvalMenuPaperSx,
      }}
      MenuListProps={{
        sx: {
          py: 1,
          minWidth: 0,
        },
      }}
    >
      {showActions ? (
        <>
          <Typography sx={menuTitleSx}>Actions</Typography>
          <MenuItem
            onClick={handleApprove}
            disabled={!options.approve}
            sx={options.approve ? menuItemSx : disabledMenuItemSx}
          >
            APPROVE
          </MenuItem>
          <MenuItem
            onClick={handleReject}
            disabled={!options.reject}
            sx={options.reject ? menuItemSx : disabledMenuItemSx}
          >
            REJECT
          </MenuItem>
          <Divider sx={{ my: 0.25 }} />
        </>
      ) : null}
      <Typography sx={menuTitleSx}>Assignments</Typography>
      <MenuItem
        onClick={handleReassign}
        disabled={!options.reassign}
        sx={options.reassign ? menuItemSx : disabledMenuItemSx}
      >
        REASSIGN
      </MenuItem>
      <MenuItem
        onClick={handleUnassign}
        disabled={!options.unassign}
        sx={options.unassign ? menuItemSx : disabledMenuItemSx}
      >
        UNASSIGN
      </MenuItem>
    </Menu>
  );
}
