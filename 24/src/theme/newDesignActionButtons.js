import { alpha } from "@mui/material/styles";

export const NEW_DESIGN_APPROVE_GREEN = "#2e7d32";
export const NEW_DESIGN_REJECT_RED = "#d32f2f";

/** Shared outlined-button hover fill for new-design APPROVE / REJECT actions. */
export const newDesignOutlinedButtonHoverSx = (borderColor, color = borderColor) => ({
  boxShadow: "none",
  fontWeight: 400,
  "&:hover": {
    bgcolor: alpha(borderColor, 0.08),
    borderColor,
    color,
    boxShadow: "none",
  },
  "&:active": {
    bgcolor: alpha(borderColor, 0.08),
    borderColor,
    color,
    boxShadow: "none",
  },
});

/** Shared outlined-button hover fill for new-design MORE / primary actions. */
export const newDesignPrimaryOutlinedButtonHoverSx = {
  boxShadow: "none",
  fontWeight: 400,
  "&:hover": {
    bgcolor: (theme) => alpha(theme.palette.primary.main, 0.08),
    borderColor: "primary.main",
    color: "primary.main",
    boxShadow: "none",
  },
  "&:active": {
    bgcolor: (theme) => alpha(theme.palette.primary.main, 0.08),
    borderColor: "primary.main",
    color: "primary.main",
    boxShadow: "none",
  },
};

export const newDesignTableApproveButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  py: 0.75,
  minWidth: 0,
  px: 1.25,
  lineHeight: 1.4,
  color: NEW_DESIGN_APPROVE_GREEN,
  borderColor: NEW_DESIGN_APPROVE_GREEN,
  ...newDesignOutlinedButtonHoverSx(NEW_DESIGN_APPROVE_GREEN),
};

export const newDesignTableRejectButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  py: 0.75,
  minWidth: 0,
  px: 1.25,
  lineHeight: 1.4,
  color: NEW_DESIGN_REJECT_RED,
  borderColor: NEW_DESIGN_REJECT_RED,
  ...newDesignOutlinedButtonHoverSx(NEW_DESIGN_REJECT_RED),
};
