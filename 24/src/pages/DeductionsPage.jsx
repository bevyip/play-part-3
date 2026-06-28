import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import { alpha, useTheme } from "@mui/material/styles";
import {
  newDesignTableApproveButtonSx,
  newDesignTableRejectButtonSx,
} from "../theme/newDesignActionButtons";
import ApprovalActionModal from "../components/ApprovalActionModal";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import TablePagination from "@mui/material/TablePagination";
import Checkbox from "@mui/material/Checkbox";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import FilterListIcon from "@mui/icons-material/FilterList";
import SortIcon from "@mui/icons-material/Sort";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import SearchIcon from "@mui/icons-material/Search";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import EditIcon from "@mui/icons-material/Edit";
import CloseIcon from "@mui/icons-material/Close";
import AddIcon from "@mui/icons-material/Add";
import InfoIcon from "@mui/icons-material/Info";
import { ArrowDropDownIcon } from "../theme/icons";
import SaveIcon from "@mui/icons-material/Save";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import ForumIcon from "@mui/icons-material/Forum";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import DeductionModal from "../components/DeductionModal";
import ViewApprovalModal from "../components/ViewApprovalModal";
import ViewAuditLogPopover from "../components/ViewAuditLogPopover";
import {
  cloneDefaultApprovalHistory,
  getAssigneeChangeHistoryEntry,
  prependApprovalHistoryEntry,
} from "../data/approvalHistory";
import {
  applyRejectManageAction,
  applyReassignManageAction,
  applyUnassignManageAction,
} from "../utils/applyApprovalManageAction";
import { infoIconSx, TABLE_TEXT_COLOR } from "../theme/colors";
import { simpleTooltipProps } from "../theme/tooltips";
import { getApprovalStatusDisplayWithColor } from "../utils/approvalHelpers";

const SUMMARY_CARDS = [
  {
    title: "Outstanding Deductions",
    value: "42",
    toggleLabel: "View All Outstanding Deductions",
  },
  {
    title: "Your Active Deductions",
    value: "3",
    toggleLabel: "View Active Deductions Assigned to You",
  },
  {
    title: "Likely Dispute Total",
    value: "$1,131.66",
    toggleLabel: "View Likely Disputable Deductions",
  },
];

const DEFAULT_ROWS = [
  {
    id: 1,
    checkNumber: "3074458",
    checkDate: "8/30/2024",
    depositDate: "",
    customer: "UNFI West",
    invoiceNumber: "SN000115323",
    retailer: "",
    reason: "Other - Hol...",
    amount: "-$880.19",
    docText: "Holiday promo deduction",
    promo: "HOL2024",
    glAccount: "4100-100",
    clearing: "No",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Unassigned",
    approvedBy: null,
  },
  {
    id: 2,
    checkNumber: "3074458",
    checkDate: "8/30/2024",
    depositDate: "",
    customer: "UNFI West",
    invoiceNumber: "SN000115323",
    retailer: "",
    reason: "Other - Hol...",
    amount: "-$320.00",
    docText: "Holiday promo deduction",
    promo: "HOL2024",
    glAccount: "4100-100",
    clearing: "No",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Unassigned",
    approvedBy: null,
  },
  {
    id: 3,
    checkNumber: "3074458",
    checkDate: "8/30/2024",
    depositDate: "",
    customer: "UNFI West",
    invoiceNumber: "SN000115323",
    retailer: "",
    reason: "Other - Hol...",
    amount: "-$740.50",
    docText: "Holiday promo deduction",
    promo: "HOL2024",
    glAccount: "4100-100",
    clearing: "No",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Unassigned",
    approvedBy: null,
  },
  {
    id: 4,
    checkNumber: "3074458",
    checkDate: "8/30/2024",
    depositDate: "",
    customer: "UNFI West",
    invoiceNumber: "SN000115323",
    retailer: "",
    reason: "Other - Hol...",
    amount: "-$1850.03",
    docText: "Holiday promo deduction",
    promo: "HOL2024",
    glAccount: "4100-100",
    clearing: "No",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Assigned to Beverly",
    approvedBy: null,
  },
  {
    id: 5,
    checkNumber: "8100701995",
    checkDate: "4/8/2025",
    depositDate: "4/7/2025",
    customer: "Albertsons / Safeway",
    invoiceNumber: "357240",
    retailer: "Shaw's (NE)",
    reason: "Distributor",
    amount: "-$450.75",
    docText: "Distributor chargeback",
    promo: "",
    glAccount: "4200-200",
    clearing: "Yes",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Assigned to Odette",
    approvedBy: "Matt",
  },
  {
    id: 6,
    checkNumber: "8100701995",
    checkDate: "4/8/2025",
    depositDate: "4/7/2025",
    customer: "Albertsons / Safeway",
    invoiceNumber: "357240",
    retailer: "Shaw's (NE)",
    reason: "Retailer - S...",
    amount: "-$1100.00",
    docText: "Retailer shortage claim",
    promo: "",
    glAccount: "4200-200",
    clearing: "Yes",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Assigned to Beverly",
    approvedBy: null,
  },
  {
    id: 7,
    checkNumber: "8100701995",
    checkDate: "4/8/2025",
    depositDate: "4/7/2025",
    customer: "Albertsons / Safeway",
    invoiceNumber: "357240",
    retailer: "Shaw's (NE)",
    reason: "Retailer - S...",
    amount: "-$210.00",
    docText: "Retailer shortage claim",
    promo: "",
    glAccount: "4200-200",
    clearing: "Yes",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Approved",
    approvedBy: "Justin Hunter",
  },
  {
    id: 8,
    checkNumber: "8100701995",
    checkDate: "4/8/2025",
    depositDate: "4/7/2025",
    customer: "Albertsons / Safeway",
    invoiceNumber: "357240",
    retailer: "Shaw's (NE)",
    reason: "Retailer - S...",
    amount: "-$62.00",
    docText: "Retailer shortage claim",
    promo: "",
    glAccount: "4200-200",
    clearing: "Yes",
    attached: false,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Unassigned",
    approvedBy: null,
  },
  {
    id: 9,
    checkNumber: "8100701995",
    checkDate: "4/8/2025",
    depositDate: "4/7/2025",
    customer: "Albertsons / Safeway",
    invoiceNumber: "357240",
    retailer: "Shaw's (NE)",
    reason: "Retailer - S...",
    amount: "-$48.50",
    docText: "Retailer shortage claim",
    promo: "",
    glAccount: "4200-200",
    clearing: "Yes",
    attached: true,
    deductionStatus: "Dispute - Filing in progress",
    approvalStatus: "Assigned to Beverly",
    approvedBy: null,
  },
  {
    id: 10,
    checkNumber: "8100701995",
    checkDate: "4/8/2025",
    depositDate: "4/7/2025",
    customer: "Albertsons / Safeway",
    invoiceNumber: "357240",
    retailer: "Shaw's (NE)",
    reason: "Retailer - S...",
    amount: "-$33.16",
    docText: "Retailer shortage claim",
    promo: "",
    glAccount: "4200-200",
    clearing: "Yes",
    attached: true,
    deductionStatus: null,
    approvalStatus: null,
    approvedBy: null,
  },
];

const cloneDefaultRows = () =>
  DEFAULT_ROWS.map((row) => ({
    ...row,
    approvalHistory: cloneDefaultApprovalHistory(),
  }));

const CURRENT_USER = "Beverly";

function getAssigneeName(approvalStatus) {
  if (!approvalStatus?.startsWith("Assigned to ")) return null;
  return approvalStatus.slice("Assigned to ".length);
}

function isAwaitingCurrentUserApproval(approvalStatus) {
  return getAssigneeName(approvalStatus) === CURRENT_USER;
}

function isAssignedToOtherUser(approvalStatus) {
  const assignee = getAssigneeName(approvalStatus);
  return Boolean(assignee && assignee !== CURRENT_USER);
}

function isUnassignedStatus(approvalStatus) {
  return !approvalStatus || approvalStatus === "Unassigned";
}

/** Table cell display — never shows "Pending Approval". */
function getApprovalStatusDisplay(approvalStatus) {
  if (approvalStatus === "Pending Approval") {
    return `Assigned to ${CURRENT_USER}`;
  }
  return approvalStatus || null;
}

/** Edit-mode dropdown placeholder derived from assignment. */
function getApprovalDropdownPlaceholder(approvalStatus) {
  if (isAwaitingCurrentUserApproval(approvalStatus)) {
    return "Awaiting your approval";
  }
  if (isAssignedToOtherUser(approvalStatus)) {
    return "Pending Approval";
  }
  return getApprovalStatusDisplay(approvalStatus) || "";
}

const COLUMNS = [
  { id: "view", label: "View", minWidth: 60, align: "center" },
  { id: "checkNumber", label: "Check Number", minWidth: 130 },
  { id: "checkDate", label: "Check Date", minWidth: 110 },
  { id: "depositDate", label: "Deposit Date", minWidth: 120 },
  { id: "customer", label: "Customer", minWidth: 180 },
  { id: "invoiceNumber", label: "Invoice Number", minWidth: 150 },
  { id: "retailer", label: "Retailer", minWidth: 140 },
  { id: "reason", label: "Reason", minWidth: 130 },
  { id: "amount", label: "Amount", minWidth: 110 },
  { id: "docText", label: "Doc Text", minWidth: 220 },
  { id: "promo", label: "Promo", minWidth: 120 },
  { id: "glAccount", label: "GL Account", minWidth: 120 },
  { id: "clearing", label: "Clearing", minWidth: 100 },
  { id: "deductionStatus", label: "Deduction Status", minWidth: 200 },
  { id: "approvalStatus", label: "Approval Status", minWidth: 190 },
  { id: "approvedBy", label: "Approved By", minWidth: 130 },
  {
    id: "approvalActions",
    label: "Approval Actions",
    minWidth: 200,
    align: "center",
    stickyRight: true,
  },
  {
    id: "edit",
    label: "Edit",
    minWidth: 70,
    align: "center",
    stickyRight: true,
  },
];

const CHECKBOX_COL_WIDTH = 52;
const VIEW_COL_WIDTH = 60;
const EDIT_COL_WIDTH = 70;
const APPROVAL_ACTIONS_COL_WIDTH = 200;
const MANAGE_COL_WIDTH = 280;

const NEW_DESIGN_ICON_GREY = "#6b7280";

const ASSIGNED_COLUMN = { id: "assigned", label: "Assigned", minWidth: 110 };

const MANAGE_COLUMN = {
  id: "manage",
  label: "Manage",
  minWidth: MANAGE_COL_WIDTH,
  align: "center",
  stickyRight: true,
};

function getNewDesignApprovalStatusDisplay(row) {
  return getApprovalStatusDisplayWithColor(row.approvalStatus);
}

function getNewDesignAssigned(row) {
  if (isUnassignedStatus(row.approvalStatus)) return "";
  return getAssigneeName(row.approvalStatus) || "";
}

function getNewDesignApprovedBy(row) {
  if (isUnassignedStatus(row.approvalStatus)) return "";
  if (row.approvedBy) return row.approvedBy;
  if (isAwaitingCurrentUserApproval(row.approvalStatus)) {
    return CURRENT_USER;
  }
  return "";
}

function getTableColumns(newDesignEnabled) {
  if (!newDesignEnabled) {
    return COLUMNS;
  }

  const columns = [];

  COLUMNS.forEach((col) => {
    if (col.id === "approvalActions" || col.id === "edit") {
      return;
    }

    columns.push(col);

    if (col.id === "approvalStatus") {
      columns.push(ASSIGNED_COLUMN);
    }
  });

  columns.push(MANAGE_COLUMN);
  return columns;
}

function getTableMinWidth(newDesignEnabled) {
  return newDesignEnabled ? 2620 : 2500;
}

function getColumnHeaderSx(col) {
  if (col.stickyLeft || col.id === "view") {
    return getStickyLeftHeadSx(CHECKBOX_COL_WIDTH, col.minWidth);
  }
  if (col.id === "approvalActions") {
    return getStickyRightHeadSx(EDIT_COL_WIDTH, APPROVAL_ACTIONS_COL_WIDTH, true);
  }
  if (col.id === "edit") {
    return getStickyRightHeadSx(0, EDIT_COL_WIDTH);
  }
  if (col.stickyRight || col.id === "manage") {
    return getStickyRightHeadSx(0, col.minWidth, true);
  }
  return { ...headerCellSx, minWidth: col.minWidth };
}

function getColumnBodySx(col, editCellSx) {
  const baseSx =
    col.stickyLeft || col.id === "view"
      ? getStickyLeftBodySx(CHECKBOX_COL_WIDTH, col.minWidth)
      : col.id === "approvalActions"
        ? getStickyRightBodySx(EDIT_COL_WIDTH, APPROVAL_ACTIONS_COL_WIDTH, true)
        : col.id === "edit"
          ? getStickyRightBodySx(0, EDIT_COL_WIDTH)
          : col.stickyRight || col.id === "manage"
            ? getStickyRightBodySx(0, col.minWidth, true)
            : bodyCellSx;

  return editCellSx(baseSx);
}

function NewDesignApprovalStatusCell({ row }) {
  const { text, color } = getNewDesignApprovalStatusDisplay(row);

  return (
    <Box component="span" sx={{ color, fontSize: 14 }}>
      {text}
    </Box>
  );
}

const DEDUCTION_STATUS_OPTIONS = [
  "",
  "Dispute - Filing in progress",
  "Dispute - Resolved",
  "Accepted",
];

const APPROVAL_ACTION_OPTIONS = [
  "Awaiting your approval",
  "Approve",
  "Reject",
];

const ASSIGN_TO_OPTIONS = [
  "Beverly",
  "Kevin",
  "Odette",
  "Matt",
  "Justin Hunter",
  "kevexternal",
];

const UNASSIGNED_DISPLAY_OPTIONS = ["Unassigned", ...ASSIGN_TO_OPTIONS];

const EDIT_ROW_FONT_SIZE = 15;

const approvalMenuItemSx = {
  fontSize: EDIT_ROW_FONT_SIZE,
  "&:hover": { cursor: "pointer" },
};

const approvalStatusFormSx = (disabled) => ({
  flex: 1,
  minWidth: 0,
  "& .MuiOutlinedInput-root": {
    fontSize: EDIT_ROW_FONT_SIZE,
    color: TABLE_TEXT_COLOR,
    bgcolor: disabled ? "#fafafa" : "#fff",
    borderRadius: 0.25,
    "& fieldset": {
      borderColor: "#e8e8e8",
    },
    "&:hover fieldset": {
      borderColor: "#d1d5db",
    },
    "&.Mui-focused fieldset": {
      borderColor: "#d1d5db",
      borderWidth: 1,
    },
    "&.Mui-disabled": {
      opacity: 1,
      "& fieldset": {
        borderColor: "#e8e8e8",
      },
    },
  },
  "& .MuiSelect-select": {
    py: 0.75,
    px: 1,
    minHeight: "auto",
    display: "flex",
    alignItems: "center",
  },
  "& .MuiSelect-icon": {
    color: "#6b7280",
    fontSize: 20,
    right: 8,
  },
  "& .Mui-disabled .MuiSelect-select": {
    WebkitTextFillColor: "#9ca3af",
  },
});

const deductionStatusSelectSx = {
  fontSize: EDIT_ROW_FONT_SIZE,
  color: TABLE_TEXT_COLOR,
  width: "100%",
  "& .MuiSelect-select": {
    py: 0.25,
    px: 0,
    pr: "22px !important",
    display: "flex",
    alignItems: "center",
  },
  "&::before, &::after": {
    display: "none",
  },
  "& .MuiSelect-icon": {
    color: "#6b7280",
    fontSize: 20,
    right: 0,
  },
};

const getEditRowSx = () => ({
  position: "relative",
  zIndex: 3,
  boxShadow:
    "0 4px 8px rgba(0, 0, 0, 0.12), 0 -4px 8px rgba(0, 0, 0, 0.12)",
  "& td": {
    borderRight: "1px solid #e8e8e8",
    borderBottom: "1px solid #d1d5db",
    bgcolor: "#fff",
    fontSize: EDIT_ROW_FONT_SIZE,
    color: TABLE_TEXT_COLOR,
  },
  "& td:last-of-type": {
    borderRight: "none",
  },
});

const getEditRowCellSx = (baseSx) => ({
  ...baseSx,
  py: 0.75,
  px: 1,
});

function DeductionStatusSelect({ value, onChange }) {
  return (
    <FormControl size="small" fullWidth sx={{ minWidth: 0 }}>
      <Select
        value={value}
        displayEmpty
        onChange={onChange}
        variant="standard"
        disableUnderline
        IconComponent={ArrowDropDownIcon}
        sx={deductionStatusSelectSx}
        renderValue={(selected) => selected || "\u00a0"}
      >
        {DEDUCTION_STATUS_OPTIONS.map((opt) => (
          <MenuItem
            key={opt || "__empty"}
            value={opt}
            sx={approvalMenuItemSx}
          >
            {opt || "\u00a0"}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}

function ApprovalStatusSelect({ placeholder, disabled, value, onChange, onInfoClick }) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 0.75,
        minWidth: 0,
        width: "100%",
      }}
    >
      <FormControl
        fullWidth
        size="small"
        disabled={disabled}
        sx={approvalStatusFormSx(disabled)}
      >
        <Select
          value={value}
          displayEmpty
          onChange={onChange}
          IconComponent={ArrowDropDownIcon}
          renderValue={(selected) =>
            selected ? (
              selected
            ) : (
              <Box component="span" sx={{ color: "#9ca3af" }}>
                {placeholder || "\u00a0"}
              </Box>
            )
          }
        >
          {APPROVAL_ACTION_OPTIONS.map((opt) => (
            <MenuItem key={opt} value={opt} sx={approvalMenuItemSx}>
              {opt}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <IconButton
        size="small"
        disableRipple
        onClick={(e) => {
          e.stopPropagation();
          onInfoClick?.();
        }}
        sx={{
          p: 0,
          flexShrink: 0,
          "&:hover": { bgcolor: "transparent" },
        }}
      >
        <InfoIcon
          sx={{
            fontSize: 20,
            ...infoIconSx,
          }}
        />
      </IconButton>
    </Box>
  );
}

function UnassignedApprovalStatusSelect({
  assignModeActive,
  assignee,
  onStartAssign,
  onCancelAssign,
  onConfirmAssign,
  onAssigneeChange,
}) {
  const canConfirm = Boolean(assignee);
  const selectOptions = assignModeActive
    ? ASSIGN_TO_OPTIONS
    : UNASSIGNED_DISPLAY_OPTIONS;

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 0.5,
        minWidth: 0,
        width: "100%",
      }}
    >
      <FormControl
        fullWidth
        size="small"
        disabled={!assignModeActive}
        sx={{ ...approvalStatusFormSx(!assignModeActive), flex: 1, minWidth: 0 }}
      >
        <Select
          value={assignModeActive ? assignee : "Unassigned"}
          displayEmpty={assignModeActive}
          onChange={onAssigneeChange}
          IconComponent={ArrowDropDownIcon}
          renderValue={(selected) =>
            assignModeActive ? (
              selected ? (
                selected
              ) : (
                <Box component="span" sx={{ color: "#9ca3af" }}>
                  {"\u00a0"}
                </Box>
              )
            ) : (
              selected || "Unassigned"
            )
          }
        >
          {selectOptions.map((opt) => (
            <MenuItem key={opt} value={opt} sx={approvalMenuItemSx}>
              {opt}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      {!assignModeActive ? (
        <IconButton
          size="small"
          disableRipple
          onClick={(e) => {
            e.stopPropagation();
            onStartAssign();
          }}
          sx={{
            ...approvalActionIconButtonSx,
            bgcolor: "primary.dark",
            color: "#fff",
          }}
        >
          <AddIcon sx={{ fontSize: 14 }} />
        </IconButton>
      ) : (
        <>
          <IconButton
            size="small"
            disableRipple
            disabled={!canConfirm}
            onClick={(e) => {
              e.stopPropagation();
              onConfirmAssign();
            }}
            sx={{
              ...approvalActionIconButtonSx,
              bgcolor: canConfirm ? "primary.dark" : "#9ca3af",
              color: "#fff",
              cursor: canConfirm ? "pointer" : "default",
              "&:hover": {
                bgcolor: canConfirm ? "primary.dark" : "#9ca3af",
                opacity: canConfirm ? 0.9 : 1,
              },
              "&.Mui-disabled": {
                bgcolor: "#9ca3af",
                color: "#fff",
                opacity: 1,
              },
            }}
          >
            <AddIcon sx={{ fontSize: 14 }} />
          </IconButton>
          <IconButton
            size="small"
            disableRipple
            onClick={(e) => {
              e.stopPropagation();
              onCancelAssign();
            }}
            sx={{
              ...approvalActionIconButtonSx,
              bgcolor: "#d32f2f",
              color: "#fff",
              "&:hover": { bgcolor: "#d32f2f", opacity: 0.9 },
            }}
          >
            <CloseIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </>
      )}
    </Box>
  );
}

const bodyCellSx = {
  fontSize: 14,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  borderBottom: "1px solid #f0f0f0",
  py: 1.25,
  whiteSpace: "nowrap",
};

const headerCellSx = {
  ...bodyCellSx,
  bgcolor: "#fff",
  borderBottom: "1px solid #e0e0e0",
};

const editIconSx = { fontSize: 20, color: "primary.main" };

const editIconButtonSx = {
  p: 0.5,
  "&:hover": { bgcolor: "transparent" },
};

const approveButtonSx = newDesignTableApproveButtonSx;
const rejectButtonSx = newDesignTableRejectButtonSx;

const approvalActionIconButtonSx = {
  p: 0,
  flexShrink: 0,
  width: 20,
  height: 20,
  minWidth: 20,
  "&:hover": { opacity: 0.9 },
};

const getStickyLeftHeadSx = (left, minWidth) => ({
  ...headerCellSx,
  position: "sticky",
  left,
  zIndex: 4,
  minWidth,
  bgcolor: "#fff",
  borderRight: "1px solid #e0e0e0",
});

const getStickyLeftBodySx = (left, minWidth) => ({
  ...bodyCellSx,
  position: "sticky",
  left,
  zIndex: 2,
  minWidth,
  bgcolor: "#fff",
  borderRight: "1px solid #e0e0e0",
});

const getStickyRightHeadSx = (right, minWidth, borderLeft = false) => ({
  ...headerCellSx,
  position: "sticky",
  right,
  zIndex: 4,
  minWidth,
  bgcolor: "#fff",
  textAlign: "center",
  ...(borderLeft && { borderLeft: "1px solid #e0e0e0" }),
});

const getStickyRightBodySx = (right, minWidth, borderLeft = false) => ({
  ...bodyCellSx,
  position: "sticky",
  right,
  zIndex: 2,
  minWidth,
  bgcolor: "#fff",
  ...(borderLeft && { borderLeft: "1px solid #e0e0e0" }),
});

const VIEW_TAB_HEIGHT = 36;

const viewTabButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  borderRadius: 0.5,
  px: 2,
  py: 0.625,
  minHeight: VIEW_TAB_HEIGHT,
  lineHeight: 1.2,
};

const searchFieldSx = {
  width: 380,
  "& .MuiOutlinedInput-root": {
    borderRadius: 50,
    bgcolor: "#fff",
    fontSize: 13,
    height: VIEW_TAB_HEIGHT,
    pl: 1.25,
    pr: 1.5,
    "& fieldset": { borderRadius: 50 },
    "&:hover .MuiOutlinedInput-notchedOutline": {
      borderColor: "rgba(0, 0, 0, 0.23)",
    },
    "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
      borderColor: "primary.main",
    },
  },
  "& .MuiOutlinedInput-input": {
    py: 0,
    fontSize: 13,
  },
};

const containedToolbarSx = {
  boxShadow: "none",
  fontSize: 13,
  fontWeight: 400,
  px: 3,
  py: 0.5,
  minHeight: 34,
  minWidth: 100,
  color: "#fff",
  bgcolor: "primary.main",
  "& .MuiButton-startIcon": { color: "#fff", mr: 0.75 },
  "&:hover": { bgcolor: "primary.main", color: "#fff", boxShadow: "none" },
};

const outlinedActionSx = {
  fontSize: 13,
  fontWeight: 400,
  px: 3,
  py: 0.5,
  minHeight: 34,
  minWidth: 100,
  bgcolor: "#fff",
  "& .MuiButton-startIcon": { mr: 0.75 },
  "&:hover": { bgcolor: "#fff", boxShadow: "none" },
};

const toolbarRowGap = 3;

const linkButtonSx = {
  fontSize: 14,
  fontWeight: 400,
  p: 0,
  minWidth: 0,
  minHeight: 0,
  lineHeight: 1,
  color: "primary.main",
  display: "inline-flex",
  alignItems: "center",
  verticalAlign: "middle",
  "& .MuiButton-startIcon": {
    margin: 0,
    marginRight: "6px",
    display: "inherit",
    alignItems: "center",
  },
  "&:hover": { bgcolor: "transparent" },
};

export default function DeductionsPage({ newDesignEnabled = false, resetKey = 0 }) {
  const theme = useTheme();
  const [selectedRow, setSelectedRow] = useState(null);
  const [tableRows, setTableRows] = useState(() => cloneDefaultRows());
  const [editingRowId, setEditingRowId] = useState(null);
  const [rowEdits, setRowEdits] = useState({});
  const [approvalModalRowId, setApprovalModalRowId] = useState(null);
  const [auditLogRowId, setAuditLogRowId] = useState(null);
  const [tableActionModal, setTableActionModal] = useState(null);
  const [viewMode, setViewMode] = useState("split");
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  const applyApprovalActionToRow = (row, action, pendingEdits) => {
    const nextRow = { ...row };
    if (pendingEdits?.deductionStatus !== undefined) {
      nextRow.deductionStatus = pendingEdits.deductionStatus || null;
    }

    if (action === "approve") {
      return {
        ...nextRow,
        approvalStatus: "Approved",
        approvedBy: CURRENT_USER,
        approvalHistory: prependApprovalHistoryEntry(row.approvalHistory, {
          action: "Approved at",
          person: CURRENT_USER,
          actor: CURRENT_USER,
        }),
      };
    }
    if (action === "reject") {
      return {
        ...nextRow,
        approvalStatus: "Rejected",
        approvedBy: null,
        approvalHistory: prependApprovalHistoryEntry(row.approvalHistory, {
          action: "Rejected at",
          person: CURRENT_USER,
          actor: CURRENT_USER,
        }),
      };
    }
    return nextRow;
  };

  const updateRowById = (rowId, getNextRow) => {
    setTableRows((prev) => {
      const row = prev.find((item) => item.id === rowId);
      if (!row) return prev;

      const nextRow = getNextRow(row);
      setSelectedRow((selected) => (selected?.id === rowId ? nextRow : selected));
      return prev.map((item) => (item.id === rowId ? nextRow : item));
    });
  };

  const applyApprovalAction = (rowId, action, pendingEdits) => {
    updateRowById(rowId, (row) =>
      applyApprovalActionToRow(row, action, pendingEdits),
    );
    setEditingRowId((currentId) => (currentId === rowId ? null : currentId));
    setRowEdits((prev) => {
      const next = { ...prev };
      delete next[rowId];
      return next;
    });
  };

  const applyManageActionToRowId = (rowId, applyFn, payload) => {
    updateRowById(rowId, (row) => applyFn(row, payload));
  };

  const applyAssignAction = (rowId, assignee, pendingEdits) => {
    if (!assignee) return;

    setTableRows((prev) =>
      prev.map((row) => {
        if (row.id !== rowId) return row;

        const nextRow = { ...row };
        if (pendingEdits?.deductionStatus !== undefined) {
          nextRow.deductionStatus = pendingEdits.deductionStatus || null;
        }

        const historyEntry = getAssigneeChangeHistoryEntry(
          assignee,
          row.approvalStatus,
          CURRENT_USER,
        );

        return {
          ...nextRow,
          approvalStatus: `Assigned to ${assignee}`,
          approvalHistory: prependApprovalHistoryEntry(
            row.approvalHistory,
            historyEntry,
          ),
        };
      }),
    );
    setEditingRowId((currentId) => (currentId === rowId ? null : currentId));
    setRowEdits((prev) => {
      const next = { ...prev };
      delete next[rowId];
      return next;
    });
  };

  const handleResetTable = () => {
    setTableRows(cloneDefaultRows());
    setSelectedRow(null);
    setEditingRowId(null);
    setRowEdits({});
    setApprovalModalRowId(null);
    setAuditLogRowId(null);
    setTableActionModal(null);
    setPage(0);
  };

  useEffect(() => {
    if (resetKey === 0) return;
    handleResetTable();
  }, [resetKey]);

  const handleResetModalRow = (rowId) => {
    const defaultRow = cloneDefaultRows().find((row) => row.id === rowId);
    if (!defaultRow) return;

    setTableRows((prev) =>
      prev.map((row) => (row.id === rowId ? { ...defaultRow } : row)),
    );
    setSelectedRow((prev) => (prev?.id === rowId ? { ...defaultRow } : prev));
  };

  const approvalModalRow = tableRows.find((row) => row.id === approvalModalRowId);
  const auditLogRow = tableRows.find((row) => row.id === auditLogRowId);

  const openTableActionModal = (rowId, actionType) => {
    setTableActionModal({ rowId, actionType });
  };

  const closeTableActionModal = () => {
    setTableActionModal(null);
  };

  const handleTableActionConfirm = (payload) => {
    if (!tableActionModal) return;
    const { rowId, actionType } = tableActionModal;

    if (actionType === "reject") {
      applyManageActionToRowId(rowId, applyRejectManageAction, payload);
    } else if (actionType === "reassign") {
      applyManageActionToRowId(rowId, applyReassignManageAction, payload);
    } else if (actionType === "unassign") {
      applyManageActionToRowId(rowId, applyUnassignManageAction, payload);
    }

    closeTableActionModal();
  };

  const handleCloseAuditLogPopover = () => {
    setAuditLogRowId(null);
  };

  const enterRowEditMode = (row) => {
    if (newDesignEnabled) return;
    setEditingRowId(row.id);
    setRowEdits((prev) => ({
      ...prev,
      [row.id]: {
        deductionStatus: row.deductionStatus || "",
        approvalAction: "",
        approvalAssignMode: false,
        approvalAssignee: "",
      },
    }));
  };

  const exitRowEditMode = () => {
    setEditingRowId(null);
  };

  const updateRowEdit = (rowId, field, value) => {
    setRowEdits((prev) => ({
      ...prev,
      [rowId]: {
        ...prev[rowId],
        [field]: value,
      },
    }));
  };

  const startUnassignedAssignMode = (rowId) => {
    setRowEdits((prev) => ({
      ...prev,
      [rowId]: {
        ...prev[rowId],
        approvalAssignMode: true,
        approvalAssignee: "",
      },
    }));
  };

  const cancelUnassignedAssignMode = (rowId) => {
    setRowEdits((prev) => ({
      ...prev,
      [rowId]: {
        ...prev[rowId],
        approvalAssignMode: false,
        approvalAssignee: "",
      },
    }));
  };

  const activeTabSx = {
    bgcolor: alpha(theme.palette.primary.main, 0.1),
    color: "primary.main",
    borderColor: "primary.main",
    "&:hover": {
      bgcolor: alpha(theme.palette.primary.main, 0.1),
      borderColor: "primary.main",
    },
  };

  const inactiveTabSx = {
    bgcolor: "#fff",
    color: "#374151",
    borderColor: "#d1d5db",
    "&:hover": { bgcolor: "#fff", borderColor: "#d1d5db" },
  };

  const outlinedToolbarSx = {
    ...outlinedActionSx,
    width: 140,
    minWidth: 140,
    whiteSpace: "nowrap",
    boxSizing: "border-box",
  };

  const bulkUpdateSx = {
    ...outlinedToolbarSx,
    color: "#9ca3af",
    "&:hover": { bgcolor: "transparent" },
  };

  const activeColumns = getTableColumns(newDesignEnabled);

  const renderApprovalStatusCell = (row, isEditing, edits) => {
    if (isEditing) {
      return isUnassignedStatus(row.approvalStatus) ? (
        <UnassignedApprovalStatusSelect
          assignModeActive={edits.approvalAssignMode ?? false}
          assignee={edits.approvalAssignee ?? ""}
          onStartAssign={() => startUnassignedAssignMode(row.id)}
          onCancelAssign={() => cancelUnassignedAssignMode(row.id)}
          onAssigneeChange={(e) =>
            updateRowEdit(row.id, "approvalAssignee", e.target.value)
          }
          onConfirmAssign={() =>
            applyAssignAction(row.id, edits.approvalAssignee, rowEdits[row.id])
          }
        />
      ) : (
        <ApprovalStatusSelect
          placeholder={getApprovalDropdownPlaceholder(row.approvalStatus)}
          disabled={!isAwaitingCurrentUserApproval(row.approvalStatus)}
          value={edits.approvalAction ?? ""}
          onInfoClick={() => setApprovalModalRowId(row.id)}
          onChange={(e) => {
            const action = e.target.value;
            if (action === "Approve") {
              applyApprovalAction(row.id, "approve", rowEdits[row.id]);
            } else if (action === "Reject") {
              applyApprovalAction(row.id, "reject", rowEdits[row.id]);
            } else {
              updateRowEdit(row.id, "approvalAction", action);
            }
          }}
        />
      );
    }

    if (newDesignEnabled) {
      return <NewDesignApprovalStatusCell row={row} />;
    }

    return getApprovalStatusDisplay(row.approvalStatus);
  };

  const renderManageCell = (row) => (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 1,
        flexWrap: "nowrap",
        width: "100%",
      }}
    >
      {isAwaitingCurrentUserApproval(row.approvalStatus) && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
          <Button
            type="button"
            disableRipple
            variant="outlined"
            size="small"
            sx={approveButtonSx}
            onClick={(e) => {
              e.stopPropagation();
              applyApprovalAction(row.id, "approve", rowEdits[row.id]);
            }}
          >
            APPROVE
          </Button>
          <Button
            type="button"
            disableRipple
            variant="outlined"
            size="small"
            sx={rejectButtonSx}
            onClick={(e) => {
              e.stopPropagation();
              openTableActionModal(row.id, "reject");
            }}
          >
            REJECT
          </Button>
        </Box>
      )}
      <Tooltip title="View Audit Log" {...simpleTooltipProps}>
        <IconButton
          size="small"
          disableRipple
          onClick={(e) => {
            e.stopPropagation();
            setAuditLogRowId(row.id);
          }}
          sx={editIconButtonSx}
        >
          <ForumIcon sx={{ fontSize: 20, color: NEW_DESIGN_ICON_GREY }} />
        </IconButton>
      </Tooltip>
      <IconButton
        size="small"
        disableRipple
        onClick={(e) => e.stopPropagation()}
        sx={editIconButtonSx}
      >
        <EditIcon sx={{ fontSize: 20, color: NEW_DESIGN_ICON_GREY }} />
      </IconButton>
    </Box>
  );

  const renderTableCellContent = (col, row, isEditing, edits) => {
    switch (col.id) {
      case "view":
        return (
          <IconButton size="small" onClick={() => setSelectedRow(row)}>
            <OpenInNewIcon sx={{ fontSize: 18, color: "#6b7280" }} />
          </IconButton>
        );
      case "checkNumber":
        return row.checkNumber;
      case "checkDate":
        return row.checkDate;
      case "depositDate":
        return row.depositDate;
      case "customer":
        return row.customer;
      case "invoiceNumber":
        return row.invoiceNumber;
      case "retailer":
        return row.retailer;
      case "reason":
        return row.reason;
      case "amount":
        return row.amount;
      case "docText":
        return row.docText;
      case "promo":
        return row.promo;
      case "glAccount":
        return row.glAccount;
      case "clearing":
        return row.clearing;
      case "deductionStatus":
        if (isEditing) {
          return (
            <DeductionStatusSelect
              value={edits.deductionStatus ?? row.deductionStatus ?? ""}
              onChange={(e) =>
                updateRowEdit(row.id, "deductionStatus", e.target.value)
              }
            />
          );
        }
        return row.deductionStatus;
      case "approvalStatus":
        return renderApprovalStatusCell(row, isEditing, edits);
      case "assigned":
        return getNewDesignAssigned(row);
      case "approvedBy":
        return newDesignEnabled
          ? getNewDesignApprovedBy(row)
          : row.approvedBy || null;
      case "approvalActions":
        return (
          isAwaitingCurrentUserApproval(row.approvalStatus) && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
              <Button
                type="button"
                disableRipple
                variant="outlined"
                size="small"
                sx={approveButtonSx}
                onClick={(e) => {
                  e.stopPropagation();
                  applyApprovalAction(row.id, "approve", rowEdits[row.id]);
                }}
              >
                APPROVE
              </Button>
              <Button
                type="button"
                disableRipple
                variant="outlined"
                size="small"
                sx={rejectButtonSx}
                onClick={(e) => {
                  e.stopPropagation();
                  applyApprovalAction(row.id, "reject", rowEdits[row.id]);
                }}
              >
                REJECT
              </Button>
            </Box>
          )
        );
      case "manage":
        return renderManageCell(row);
      case "edit":
        return (
          <IconButton
            size="small"
            disableRipple
            onClick={() => {
              if (isEditing) {
                exitRowEditMode();
              } else {
                enterRowEditMode(row);
              }
            }}
            sx={editIconButtonSx}
          >
            {isEditing ? (
              <CloseIcon sx={{ fontSize: 20, color: "#6b7280" }} />
            ) : (
              <EditIcon sx={editIconSx} />
            )}
          </IconButton>
        );
      default:
        return null;
    }
  };

  return (
    <Box sx={{ bgcolor: "#f5f5f5", minHeight: "100%" }}>
      {/* Summary cards */}
      <Box
        sx={{
          display: "flex",
          gap: 3,
          p: 4,
          pb: 3,
          justifyContent: "flex-start",
        }}
      >
        {SUMMARY_CARDS.map((card) => (
          <Card
            key={card.title}
            elevation={1}
            sx={{
              width: 320,
              minHeight: 200,
              flexShrink: 0,
              borderRadius: 1,
              display: "flex",
            }}
          >
            <CardContent
              sx={{
                p: 3.5,
                flex: 1,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                "&:last-child": { pb: 3.5 },
              }}
            >
              <Typography
                sx={{
                  fontSize: 18,
                  fontWeight: 500,
                  color: "#374151",
                  lineHeight: 1.3,
                }}
              >
                {card.title}
              </Typography>
              <Typography
                sx={{
                  fontSize: 52,
                  fontWeight: 500,
                  color: "#111827",
                  my: 2,
                  lineHeight: 1,
                }}
              >
                {card.value}
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                <Switch
                  disableRipple
                  sx={{ "&:hover": { bgcolor: "transparent" } }}
                />
                <Typography
                  sx={{ fontSize: 15, color: "#6b7280", lineHeight: 1.4 }}
                >
                  {card.toggleLabel}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        ))}
      </Box>

      {/* Toolbar */}
      <Box sx={{ px: 4, pb: 2 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mb: toolbarRowGap,
          }}
        >
          <Box sx={{ display: "flex", gap: 1.5 }}>
            <Button
              disableRipple
              variant="outlined"
              size="small"
              color="primary"
              onClick={() => setViewMode("split")}
              sx={{
                ...viewTabButtonSx,
                ...(viewMode === "split" ? activeTabSx : inactiveTabSx),
              }}
            >
              VIEW SPLIT LINES
            </Button>
            <Button
              disableRipple
              variant="outlined"
              size="small"
              color="primary"
              onClick={() => setViewMode("grouped")}
              sx={{
                ...viewTabButtonSx,
                ...(viewMode === "grouped" ? activeTabSx : inactiveTabSx),
              }}
            >
              VIEW GROUPED LINES
            </Button>
          </Box>
          <TextField
            size="small"
            placeholder="Search Doc Text, Invoice #, Check #, Prom..."
            sx={searchFieldSx}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon sx={{ fontSize: 18, color: "#9ca3af" }} />
                </InputAdornment>
              ),
            }}
          />
        </Box>

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
          }}
        >
          <Button
            disableRipple
            variant="contained"
            color="primary"
            startIcon={<FilterListIcon sx={{ fontSize: 18 }} />}
            size="small"
            sx={containedToolbarSx}
          >
            FILTER
          </Button>
          <Button
            disableRipple
            variant="contained"
            color="primary"
            startIcon={<SortIcon sx={{ fontSize: 18 }} />}
            size="small"
            sx={containedToolbarSx}
          >
            SORT
          </Button>
          <Button
            disableRipple
            variant="outlined"
            color="primary"
            startIcon={<ViewColumnIcon sx={{ fontSize: 18 }} />}
            size="small"
            sx={outlinedToolbarSx}
          >
            COLUMNS
          </Button>
          <Box sx={{ flex: 1 }} />
          <Button
            disableRipple
            variant="outlined"
            size="small"
            disabled
            sx={bulkUpdateSx}
          >
            BULK UPDATE
          </Button>
        </Box>

        <Box sx={{ pt: 6 }}>
          <Button
            disableRipple
            color="primary"
            startIcon={<SaveIcon sx={{ fontSize: 20, display: "block" }} />}
            size="medium"
            sx={linkButtonSx}
          >
            SAVE VIEW
          </Button>
        </Box>
      </Box>

      {/* Table panel */}
      <Box
        sx={{
          mx: 4,
          mb: 4,
          pt: 2,
          borderTop: "1px solid",
          borderColor: "divider",
        }}
      >
        <Box
          sx={{
            bgcolor: "#fff",
            border: "1px solid #e0e0e0",
            borderRadius: 1,
            overflow: "hidden",
          }}
        >
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "flex-end",
              px: 2.5,
              py: 1.5,
            }}
          >
            <Button
              disableRipple
              color="primary"
              startIcon={
                <FileDownloadIcon sx={{ fontSize: 20, display: "block" }} />
              }
              size="medium"
              sx={linkButtonSx}
            >
              DOWNLOAD CSV
            </Button>
          </Box>

          <TableContainer sx={{ overflowX: "auto" }}>
            <Table
              stickyHeader
              size="small"
              sx={{
                minWidth: getTableMinWidth(newDesignEnabled),
                color: TABLE_TEXT_COLOR,
                "& .MuiTableCell-root": { color: TABLE_TEXT_COLOR },
              }}
            >
              <TableHead
                sx={{
                  "& .MuiTableCell-head": {
                    bgcolor: "#fff",
                    color: TABLE_TEXT_COLOR,
                  },
                }}
              >
                <TableRow>
                  <TableCell
                    padding="checkbox"
                    sx={getStickyLeftHeadSx(0, CHECKBOX_COL_WIDTH)}
                  >
                    <Checkbox size="small" />
                  </TableCell>
                  {activeColumns.map((col) => (
                    <TableCell
                      key={col.id}
                      align={col.align || "left"}
                      sx={getColumnHeaderSx(col)}
                    >
                      {col.label}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {tableRows.map((row) => {
                  const isEditing = editingRowId === row.id;
                  const edits = rowEdits[row.id] || {};
                  const editCellSx = (baseSx) =>
                    isEditing ? getEditRowCellSx(baseSx) : baseSx;

                  return (
                    <TableRow
                      key={row.id}
                      hover={!isEditing}
                      sx={{
                        ...(isEditing ? getEditRowSx() : {}),
                        ...(!isEditing && {
                          "&:last-child td": { borderBottom: 0 },
                        }),
                      }}
                    >
                      <TableCell
                        padding="checkbox"
                        sx={editCellSx(
                          getStickyLeftBodySx(0, CHECKBOX_COL_WIDTH),
                        )}
                      >
                        <Checkbox size="small" />
                      </TableCell>
                      {activeColumns.map((col) => (
                        <TableCell
                          key={col.id}
                          align={col.align || "left"}
                          sx={{
                            ...getColumnBodySx(col, editCellSx),
                            ...(col.id === "approvalStatus" &&
                              isEditing && {
                                position: "relative",
                                "&::before": {
                                  content: '""',
                                  position: "absolute",
                                  inset: 0,
                                  border: "1px solid",
                                  borderColor: (theme) =>
                                    alpha(theme.palette.primary.main, 0.55),
                                  pointerEvents: "none",
                                  zIndex: 1,
                                },
                              }),
                          }}
                          onDoubleClick={
                            !newDesignEnabled &&
                            col.id === "approvalStatus" &&
                            !isEditing
                              ? () => enterRowEditMode(row)
                              : undefined
                          }
                        >
                          {renderTableCellContent(col, row, isEditing, edits)}
                        </TableCell>
                      ))}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>

          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              borderTop: "1px solid #e0e0e0",
              pl: 2,
              pr: 1,
            }}
          >
            <TablePagination
              component="div"
              count={tableRows.length}
              page={page}
              onPageChange={(_, p) => setPage(p)}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setPage(0);
              }}
              rowsPerPageOptions={[25, 50]}
              sx={{
                flex: 1,
                ml: 1,
                borderTop: 0,
                fontSize: 13,
                color: TABLE_TEXT_COLOR,
                "& .MuiTablePagination-toolbar": {
                  minHeight: 52,
                  pl: 0,
                },
                "& .MuiTablePagination-selectLabel, & .MuiTablePagination-displayedRows":
                  { color: TABLE_TEXT_COLOR },
              }}
            />
          </Box>
        </Box>
      </Box>

      <ApprovalActionModal
        open={Boolean(tableActionModal)}
        actionType={tableActionModal?.actionType ?? null}
        onClose={closeTableActionModal}
        onConfirm={handleTableActionConfirm}
      />

      {selectedRow && (
        <DeductionModal
          open={!!selectedRow}
          onClose={() => setSelectedRow(null)}
          newDesignEnabled={newDesignEnabled}
          approvalStatus={selectedRow.approvalStatus}
          approvalHistory={selectedRow.approvalHistory}
          onApprove={() => {
            applyApprovalAction(selectedRow.id, "approve");
          }}
          onReject={(payload) => {
            applyManageActionToRowId(
              selectedRow.id,
              applyRejectManageAction,
              payload,
            );
          }}
          onReassign={(payload) => {
            const normalizedPayload =
              typeof payload === "string" ? { assignee: payload } : payload;
            applyManageActionToRowId(
              selectedRow.id,
              applyReassignManageAction,
              normalizedPayload,
            );
          }}
          onUnassign={(payload) => {
            applyManageActionToRowId(
              selectedRow.id,
              applyUnassignManageAction,
              payload,
            );
          }}
          onReset={() => handleResetModalRow(selectedRow.id)}
        />
      )}

      <ViewAuditLogPopover
        open={auditLogRowId !== null}
        onClose={handleCloseAuditLogPopover}
        approvalHistory={auditLogRow?.approvalHistory}
      />

      <ViewApprovalModal
        open={approvalModalRowId !== null}
        onClose={() => setApprovalModalRowId(null)}
        approvalStatus={approvalModalRow?.approvalStatus}
        approvalHistory={approvalModalRow?.approvalHistory}
        canApprove={
          approvalModalRow
            ? isAwaitingCurrentUserApproval(approvalModalRow.approvalStatus)
            : false
        }
        onApprove={() =>
          approvalModalRow &&
          applyApprovalAction(
            approvalModalRow.id,
            "approve",
            rowEdits[approvalModalRow.id],
          )
        }
        onReject={() =>
          approvalModalRow &&
          applyApprovalAction(
            approvalModalRow.id,
            "reject",
            rowEdits[approvalModalRow.id],
          )
        }
        onReassign={({ to }) => {
          if (!approvalModalRow || !to.length) return;
          const assignee = to[0];
          const previousStatus = approvalModalRow.approvalStatus;
          const nextStatus = `Assigned to ${assignee}`;
          const historyEntry = getAssigneeChangeHistoryEntry(
            assignee,
            previousStatus,
            CURRENT_USER,
          );
          const nextHistory = prependApprovalHistoryEntry(
            approvalModalRow.approvalHistory,
            historyEntry,
          );
          setTableRows((prev) =>
            prev.map((row) =>
              row.id === approvalModalRow.id
                ? {
                    ...row,
                    approvalStatus: nextStatus,
                    approvalHistory: nextHistory,
                  }
                : row,
            ),
          );
        }}
      />
    </Box>
  );
}
