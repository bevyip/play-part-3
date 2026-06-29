import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import ArrowDropUpIcon from "@mui/icons-material/ArrowDropUp";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";

export {
  ArrowDropDownIcon,
  ArrowDropUpIcon,
  KeyboardArrowDownIcon,
  KeyboardArrowUpIcon,
};

/** Chevron sizes for buttons and panel toggles. */
export const BUTTON_CHEVRON_FONT_SIZE = 22;
export const BUTTON_CHEVRON_FONT_SIZE_LG = 26;
export const BUTTON_CHEVRON_FONT_SIZE_SM = 18;

export const buttonChevronSx = { fontSize: BUTTON_CHEVRON_FONT_SIZE };
export const buttonChevronLgSx = { fontSize: BUTTON_CHEVRON_FONT_SIZE_LG };
export const buttonChevronSmSx = { fontSize: BUTTON_CHEVRON_FONT_SIZE_SM };

/** Shared width for MORE / MANAGE approval action dropdown menus. */
export const APPROVAL_MENU_WIDTH = 116;

export const approvalMenuPaperSx = {
  mt: 0.5,
  width: APPROVAL_MENU_WIDTH,
  minWidth: `${APPROVAL_MENU_WIDTH}px !important`,
  maxWidth: APPROVAL_MENU_WIDTH,
  borderRadius: 1,
  boxShadow: "0 4px 12px rgba(0, 0, 0, 0.12)",
};
