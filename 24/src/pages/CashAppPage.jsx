import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import InputAdornment from "@mui/material/InputAdornment";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import FormControl from "@mui/material/FormControl";
import FormHelperText from "@mui/material/FormHelperText";
import InputLabel from "@mui/material/InputLabel";
import IconButton from "@mui/material/IconButton";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import LinearProgress from "@mui/material/LinearProgress";
import RemoveIcon from "@mui/icons-material/Remove";
import AddIcon from "@mui/icons-material/Add";
import RefreshIcon from "@mui/icons-material/Refresh";
import OpenInFullIcon from "@mui/icons-material/OpenInFull";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import SearchIcon from "@mui/icons-material/Search";
import InfoIcon from "@mui/icons-material/Info";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import EventIcon from "@mui/icons-material/Event";
import { ArrowDropDownIcon, ArrowDropUpIcon, buttonChevronLgSx, buttonChevronSmSx } from "../theme/icons";
import KeyboardArrowRightIcon from "@mui/icons-material/KeyboardArrowRight";
import DeleteIcon from "@mui/icons-material/Delete";
import GetAppIcon from "@mui/icons-material/GetApp";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import FilterListIcon from "@mui/icons-material/FilterList";
import DensityMediumIcon from "@mui/icons-material/DensityMedium";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import { infoIconSx, TABLE_TEXT_COLOR } from "../theme/colors";
import ViewApprovalModal from "../components/ViewApprovalModal";
import ViewAuditLogPopover from "../components/ViewAuditLogPopover";
import MiniApprovalComponent from "../components/MiniApprovalComponent";
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
import { alpha, useTheme } from "@mui/material/styles";
import { useEffect, useRef, useState } from "react";

const PURPLE = "#9c27b0";
const LABEL_COLOR = "#6b7280";
const HEADER_TEXT_COLOR = "#1f2937";
const FIELD_FONT_SIZE = 14;
const LABEL_FONT_SIZE = 13;
const RIGHT_FIELD_FONT_SIZE = 15;
const RIGHT_LABEL_FONT_SIZE = 14;
const ICON_MUTED = "#6b7280";

const pressableResetSx = {
  "&:active": { boxShadow: "none", transform: "none" },
  "&.Mui-focusVisible": { outline: "none" },
};

const iconButtonFlatSx = {
  ...pressableResetSx,
  "&:hover": { bgcolor: "transparent" },
  "&:active": { bgcolor: "transparent" },
  "&.Mui-focusVisible": { bgcolor: "transparent" },
};

const textButtonFlatSx = {
  ...pressableResetSx,
  "&:hover": { bgcolor: "transparent" },
  "&:active": { bgcolor: "transparent" },
};

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
  "&:hover": { bgcolor: "transparent", color: "primary.main" },
  "&:active": { bgcolor: "transparent", color: "primary.main" },
};

const containedFlatSx = {
  boxShadow: "none",
  ...pressableResetSx,
  "&:hover": { boxShadow: "none", bgcolor: "primary.main" },
  "&:active": { boxShadow: "none", bgcolor: "primary.main" },
  "&.Mui-focusVisible": { bgcolor: "primary.main" },
};

const outlinedFlatSx = (borderColor, color, hoverBg = "transparent") => ({
  ...pressableResetSx,
  "&:hover": { borderColor, color, bgcolor: hoverBg },
  "&:active": { borderColor, color, bgcolor: hoverBg },
});

const lineNavDisabledButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  color: "#9ca3af",
  borderColor: "#e0e0e0",
  "&.Mui-disabled": {
    color: "#9ca3af",
    borderColor: "#e0e0e0",
  },
};

const lineNavActiveButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  ...outlinedFlatSx("primary.main", "primary.main", "#fff"),
};

const DETAIL_HEADER_CONTROL_HEIGHT = 30;

const DETAIL_PANEL_MAX_HEIGHT = {
  collapsed: { default: 380, newDesign: 460 },
  expanded: { default: 560, newDesign: 640 },
};

const detailHeaderControlTextSx = {
  fontSize: 13,
  fontWeight: 400,
  lineHeight: 1.2,
};

const detailHeaderSplitButtonSx = {
  ...detailHeaderControlTextSx,
  py: 0.25,
  minHeight: DETAIL_HEADER_CONTROL_HEIGHT,
  boxSizing: "border-box",
  ...outlinedFlatSx("primary.main", "primary.main", "#fff"),
};

const APPROVAL_LAYOUT_OPTIONS = [
  { id: "first", label: "FIRST", final: false },
  { id: "final", label: "FINAL", final: true },
];

const APPROVAL_LAYOUT_BORDER = "#000";
const APPROVAL_LAYOUT_ACTIVE_BG = "#424242";

const approvalLayoutSegmentSx = (active) => ({
  ...detailHeaderControlTextSx,
  border: "none",
  m: 0,
  px: 1.5,
  height: "100%",
  minWidth: 56,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  fontFamily: "inherit",
  boxSizing: "border-box",
  color: active ? "#fff" : HEADER_TEXT_COLOR,
  bgcolor: active ? APPROVAL_LAYOUT_ACTIVE_BG : "transparent",
  transition: "background-color 0.15s ease, color 0.15s ease",
  "&:focus-visible": {
    outline: "2px solid",
    outlineColor: APPROVAL_LAYOUT_BORDER,
    outlineOffset: -2,
  },
});

function ApprovalLayoutSegmentedToggle({ approvalLayoutFinal, onChange }) {
  return (
    <Box
      role="group"
      aria-label="Approval layout"
      sx={{
        display: "inline-flex",
        alignItems: "stretch",
        height: DETAIL_HEADER_CONTROL_HEIGHT,
        minHeight: DETAIL_HEADER_CONTROL_HEIGHT,
        border: "1px solid",
        borderColor: APPROVAL_LAYOUT_BORDER,
        borderRadius: 1,
        overflow: "hidden",
        flexShrink: 0,
        bgcolor: "#fff",
        boxSizing: "border-box",
      }}
    >
      {APPROVAL_LAYOUT_OPTIONS.map((option, index) => {
        const isActive = approvalLayoutFinal === option.final;

        return (
          <Box
            key={option.id}
            component="button"
            type="button"
            aria-pressed={isActive}
            onClick={() => onChange?.(option.final)}
            sx={{
              ...approvalLayoutSegmentSx(isActive),
              borderLeft: index > 0 ? "1px solid" : "none",
              borderColor: index > 0 ? APPROVAL_LAYOUT_BORDER : undefined,
            }}
          >
            {option.label}
          </Box>
        );
      })}
    </Box>
  );
}

const filledIconFlatSx = (bg) => ({
  ...pressableResetSx,
  "&:hover": { bgcolor: bg },
  "&:active": { bgcolor: bg },
  "&.Mui-focusVisible": { bgcolor: bg },
});

const docToolIconButtonSx = {
  bgcolor: "#e8e8e8",
  color: "#4b5563",
  width: 38,
  height: 38,
  ...filledIconFlatSx("#e8e8e8"),
};

const tableBodyCellSx = {
  fontSize: 13,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  borderBottom: "1px solid #f0f0f0",
  py: 1.75,
  px: 1.5,
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
};

const tableHeadCellSx = {
  ...tableBodyCellSx,
  borderBottom: "1px solid #e0e0e0",
  bgcolor: "#fff",
};

const rightTableHeadCellSx = { ...tableHeadCellSx, fontSize: 14 };
const rightTableBodyCellSx = { ...tableBodyCellSx, fontSize: 14 };

const rightColumnInputSx = {
  "& .MuiInputBase-input": { fontSize: RIGHT_FIELD_FONT_SIZE },
  "& .MuiInputLabel-root": {
    fontSize: RIGHT_LABEL_FONT_SIZE,
    color: LABEL_COLOR,
  },
  "& .MuiInputLabel-shrink": { fontSize: RIGHT_LABEL_FONT_SIZE },
  "& .MuiSelect-select": { fontSize: RIGHT_FIELD_FONT_SIZE },
  "& .MuiSelect-icon": { fontSize: 22, color: ICON_MUTED },
  "& .MuiInput-underline:hover:not(.Mui-disabled):before": {
    borderBottomColor: "#e0e0e0",
  },
  "& .MuiOutlinedInput-root:hover .MuiOutlinedInput-notchedOutline": {
    borderColor: "#d1d5db",
  },
};

const rightPanelFieldSx = {
  "& .MuiInputBase-input": { fontSize: RIGHT_FIELD_FONT_SIZE, py: 0.5 },
  "& .MuiInputLabel-root": {
    fontSize: RIGHT_LABEL_FONT_SIZE,
    color: LABEL_COLOR,
  },
  "& .MuiInputLabel-shrink": { fontSize: RIGHT_LABEL_FONT_SIZE },
  "& .MuiSelect-select": {
    fontSize: RIGHT_FIELD_FONT_SIZE,
    py: 0.5,
    pr: "24px !important",
  },
  "& .MuiSelect-icon": { fontSize: 22, color: ICON_MUTED },
};

const CHECK_TABLE_COLUMNS = [
  { id: "invoiceNumber", label: "Invoice Number", minWidth: 160 },
  { id: "discountAmount", label: "Discount Amount", minWidth: 140 },
  { id: "invoiceDate", label: "Invoice Date", minWidth: 120 },
  { id: "remarks", label: "Remarks", minWidth: 140 },
  { id: "grossAmount", label: "Gross Amount", minWidth: 130 },
  { id: "checkDate", label: "Check Date", minWidth: 120 },
];

const CHECK_TABLE_MIN_WIDTH = CHECK_TABLE_COLUMNS.reduce(
  (sum, col) => sum + col.minWidth,
  0,
);

const CHECK_TABLE_ROWS = [
  {
    invoiceNumber: "17043",
    discountAmount: "799.83",
    invoiceDate: "11/01/2024",
    remarks: "CONFIDO F...",
    grossAmount: "36694.64",
    checkDate: "11/25/2024",
    highlight: true,
  },
  {
    invoiceNumber: "ALW013973208-01",
    discountAmount: "0.00",
    invoiceDate: "11/21/2024",
    remarks: "CONFIDO F...",
    grossAmount: "-6143.37",
    checkDate: "11/25/2024",
  },
  {
    invoiceNumber: "ALW013857654-01",
    discountAmount: "0.00",
    invoiceDate: "11/21/2024",
    remarks: "CONFIDO F...",
    grossAmount: "-3887.00",
    checkDate: "11/25/2024",
  },
  {
    invoiceNumber: "ALW013807263-01",
    discountAmount: "0.00",
    invoiceDate: "11/21/2024",
    remarks: "CONFIDO F...",
    grossAmount: "-1271.82",
    checkDate: "11/25/2024",
  },
  {
    invoiceNumber: "ALW013807999-01",
    discountAmount: "0.00",
    invoiceDate: "11/21/2024",
    remarks: "CONFIDO F...",
    grossAmount: "-4984.04",
    checkDate: "11/25/2024",
  },
];

const LINE_ITEM_ROWS = [
  {
    line: "1",
    invoiceNumber: "17043",
    amount: "$35,894.81",
    type: "Deduction",
    expandable: false,
  },
  {
    line: "2",
    invoiceNumber: "ALW013973208-01",
    amount: "",
    type: "",
    expandable: true,
  },
  {
    line: "3",
    invoiceNumber: "ALW013857654-01",
    amount: "$-3,887.00",
    type: "Deduction",
    expandable: false,
  },
  {
    line: "4",
    invoiceNumber: "ALW013807263-01",
    amount: "$-1,271.82",
    type: "Deduction",
    expandable: false,
  },
];

/** Demo line items with detail panels — Back/Next cycle through these only. */
const NAVIGABLE_LINE_ITEMS = ["1", "3", "4"];

const CURRENT_USER = "Beverly";

const APPROVAL_ACTION_OPTIONS = ["Awaiting your approval", "Approve", "Reject"];

const UNASSIGNED_ASSIGNEE_OPTIONS = [
  "Beverly",
  "Odette",
  "Matt",
  "Justin Hunter",
  "Kevin",
];

const UNASSIGNED_ACTION_OPTIONS = [
  "Unassigned",
  ...UNASSIGNED_ASSIGNEE_OPTIONS,
];

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

function getApprovalStatusFieldValue(approvalStatus) {
  if (isAwaitingCurrentUserApproval(approvalStatus)) {
    return "Awaiting your approval";
  }
  if (isAssignedToOtherUser(approvalStatus)) {
    return "Pending Approval";
  }
  if (isUnassignedStatus(approvalStatus)) {
    return "Unassigned";
  }
  return approvalStatus || "";
}

const LINE_ITEM_DETAILS = {
  1: {
    type: "Deduction",
    netAmount: "-$2,891.28",
    invoiceNumber: "Q642019",
    approvalStatus: "Assigned to Beverly",
    deductionStatus: "",
    clearing: "No",
    reason: "Distributer - Expired at DC",
    glAccount: "",
    planningGroup: "",
    productFamily: "",
    promo: "",
    description: "Distributor - Expired at DC_",
    memo: "",
    executionStartDate: "",
    executionEndDate: "",
    invoiceDate: "04/14/2025",
    dc: "",
    rate: "$1.33",
    quantity: "1,224",
    class: "",
  },
  3: {
    type: "Deduction",
    netAmount: "-$3,887.00",
    invoiceNumber: "ALW013857654-01",
    approvalStatus: "Assigned to Odette",
    deductionStatus: "",
    clearing: "No",
    reason: "Distributer - Expired at DC",
    glAccount: "",
    planningGroup: "",
    productFamily: "",
    promo: "",
    description: "Distributor - Short shipment credit",
    memo: "",
    executionStartDate: "",
    executionEndDate: "",
    invoiceDate: "11/21/2024",
    dc: "",
    rate: "",
    quantity: "",
    class: "",
  },
  4: {
    type: "Deduction",
    netAmount: "-$1,271.82",
    invoiceNumber: "ALW013807263-01",
    approvalStatus: "Unassigned",
    deductionStatus: "",
    clearing: "No",
    reason: "Distributer - Expired at DC",
    glAccount: "",
    planningGroup: "",
    productFamily: "",
    promo: "",
    description: "Distributor - Damaged goods allowance",
    memo: "",
    executionStartDate: "",
    executionEndDate: "",
    invoiceDate: "11/21/2024",
    dc: "",
    rate: "",
    quantity: "",
    class: "",
  },
};

function cloneDefaultLineItemDetails() {
  return Object.fromEntries(
    Object.entries(LINE_ITEM_DETAILS).map(([line, detail]) => [
      line,
      {
        ...detail,
        approvedBy: null,
        approvalHistory: cloneDefaultApprovalHistory(),
      },
    ]),
  );
}

function applyLineItemApprovalAction(detail, action, assignee) {
  if (action === "approve") {
    return {
      ...detail,
      approvalStatus: "Approved",
      approvedBy: CURRENT_USER,
      approvalHistory: prependApprovalHistoryEntry(detail.approvalHistory, {
        action: "Approved at",
        person: CURRENT_USER,
      }),
    };
  }
  if (action === "reject") {
    return {
      ...detail,
      approvalStatus: "Rejected",
      approvedBy: null,
      approvalHistory: prependApprovalHistoryEntry(detail.approvalHistory, {
        action: "Rejected at",
        person: CURRENT_USER,
      }),
    };
  }
  if (action === "assign" && assignee) {
    const historyEntry = getAssigneeChangeHistoryEntry(
      assignee,
      detail.approvalStatus,
      CURRENT_USER,
    );
    return {
      ...detail,
      approvalStatus: `Assigned to ${assignee}`,
      approvalHistory: prependApprovalHistoryEntry(
        detail.approvalHistory,
        historyEntry,
      ),
    };
  }
  if (action === "unassign") {
    const historyEntry = getAssigneeChangeHistoryEntry(
      "Unassign",
      detail.approvalStatus,
      CURRENT_USER,
    );
    return {
      ...detail,
      approvalStatus: "Unassigned",
      approvalHistory: prependApprovalHistoryEntry(
        detail.approvalHistory,
        historyEntry,
      ),
    };
  }
  return detail;
}

const LEFT_TOOLBAR_ITEMS = [
  { label: "COLUMNS", icon: ViewColumnIcon },
  { label: "FILTERS", icon: FilterListIcon },
  { label: "DENSITY", icon: DensityMediumIcon },
  { label: "EXPORT", icon: FileDownloadIcon },
];

const standardFieldSx = {
  "& .MuiInputBase-input": { fontSize: FIELD_FONT_SIZE, py: 0.5 },
  "& .MuiInputLabel-root": { fontSize: LABEL_FONT_SIZE, color: LABEL_COLOR },
  "& .MuiInputLabel-shrink": { fontSize: LABEL_FONT_SIZE },
  "& .MuiInput-underline:before": { borderBottomColor: "#e0e0e0" },
  "& .MuiInput-underline:hover:not(.Mui-disabled):before": {
    borderBottomColor: "#bdbdbd",
  },
  "& .MuiInput-underline:after": { borderBottomColor: "primary.main" },
};

const disabledFieldSx = {
  "& .MuiInputBase-root.Mui-disabled:before": {
    borderBottomStyle: "solid",
    borderBottomColor: "#e0e0e0",
  },
  "& .MuiInputBase-input.Mui-disabled": {
    WebkitTextFillColor: "#9ca3af",
    color: "#9ca3af",
    cursor: "default",
  },
  "& .MuiInputLabel-root.Mui-disabled": { color: LABEL_COLOR },
};

const standardSelectSx = {
  ...standardFieldSx,
  "& .MuiSelect-select": {
    fontSize: FIELD_FONT_SIZE,
    py: 0.5,
    pr: "24px !important",
  },
  "& .MuiSelect-icon": { fontSize: 20, color: ICON_MUTED },
};

const dateFieldSx = {
  "& .MuiInputBase-root": {
    alignItems: "flex-end",
  },
  "& .MuiInputBase-input": {
    fontSize: FIELD_FONT_SIZE,
    py: 0,
    pb: "4px",
    lineHeight: 1.5,
  },
  "& .MuiInputAdornment-positionEnd": {
    marginBottom: "4px",
    marginLeft: 0,
    height: "auto",
    maxHeight: "none",
    alignSelf: "flex-end",
  },
};

const dateInputProps = {
  endAdornment: (
    <InputAdornment position="end">
      <EventIcon sx={{ fontSize: 18, color: ICON_MUTED, display: "block" }} />
    </InputAdornment>
  ),
};

const fieldHelperTextSx = {
  fontSize: 12,
  color: LABEL_COLOR,
  mx: 0,
  mt: 0.5,
  lineHeight: 1.4,
};

const dottedUnderlineSx = {
  "& .MuiInput-underline:before": {
    borderBottomStyle: "dotted",
    borderBottomColor: "#d1d5db",
  },
  "& .MuiInput-underline:hover:not(.Mui-disabled):before": {
    borderBottomStyle: "dotted",
    borderBottomColor: "#d1d5db",
  },
  "& .MuiInput-underline:after": {
    borderBottomStyle: "dotted",
  },
  "& .MuiInputBase-root.Mui-disabled:before": {
    borderBottomStyle: "dotted",
    borderBottomColor: "#d1d5db",
  },
};

const infoAdornment = (
  <InputAdornment position="end">
    <InfoOutlinedIcon
      sx={{ fontSize: 18, color: ICON_MUTED, display: "block" }}
    />
  </InputAdornment>
);

const perUnitAdornment = (
  <InputAdornment position="end">
    <Box
      sx={{
        bgcolor: "#f3f4f6",
        borderRadius: 10,
        px: 1.25,
        py: 0.35,
        fontSize: 12,
        color: ICON_MUTED,
        lineHeight: 1.2,
        mb: 0.5,
      }}
    >
      Per Unit
    </Box>
  </InputAdornment>
);

const detailFormRowSx = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 2,
  mb: 2,
};

const detailFormRowFullSx = { mb: 2 };

function DetailExpandToggle({ onClick }) {
  return (
    <IconButton
      disableRipple
      onClick={onClick}
      aria-label="Expand detail panel"
      sx={{
        position: "absolute",
        right: 16,
        bottom: -18,
        width: 36,
        height: 36,
        bgcolor: "primary.main",
        color: "#fff",
        p: 0,
        zIndex: 3,
        ...filledIconFlatSx("primary.main"),
      }}
    >
      <Box
        sx={{
          width: 20,
          height: 20,
          borderRadius: "50%",
          border: "1.5px solid #fff",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <ArrowDropDownIcon sx={buttonChevronSmSx} />
      </Box>
    </IconButton>
  );
}

function StandardSelect({
  label,
  value,
  options,
  sx,
  displayEmpty = false,
  helperText,
  disabled,
}) {
  const hasValue = Boolean(value);
  return (
    <FormControl
      variant="standard"
      fullWidth
      disabled={disabled}
      sx={{ ...standardSelectSx, ...(disabled ? disabledFieldSx : {}), ...sx }}
    >
      <InputLabel shrink={hasValue}>{label}</InputLabel>
      <Select
        value={value}
        label={label}
        displayEmpty={displayEmpty}
        IconComponent={ArrowDropDownIcon}
        renderValue={(selected) => selected || "\u00a0"}
      >
        {options.map((opt) => (
          <MenuItem
            key={opt || "__empty"}
            value={opt}
            sx={{ fontSize: FIELD_FONT_SIZE }}
          >
            {opt || "\u00a0"}
          </MenuItem>
        ))}
      </Select>
      {helperText ? (
        <FormHelperText sx={fieldHelperTextSx}>{helperText}</FormHelperText>
      ) : null}
    </FormControl>
  );
}

function StandardTextField({
  label,
  value,
  sx,
  InputProps,
  disabled,
  helperText,
  ...rest
}) {
  return (
    <TextField
      label={label}
      value={value}
      variant="standard"
      fullWidth
      disabled={disabled}
      InputProps={InputProps}
      helperText={helperText}
      FormHelperTextProps={{ sx: fieldHelperTextSx }}
      sx={{ ...standardFieldSx, ...(disabled ? disabledFieldSx : {}), ...sx }}
      {...rest}
    />
  );
}

function CashAppApprovalStatusSelect({
  approvalStatus,
  actionValue,
  onActionChange,
}) {
  const awaiting = isAwaitingCurrentUserApproval(approvalStatus);
  const unassigned = isUnassignedStatus(approvalStatus);
  const interactive = awaiting || unassigned;
  const formControlRef = useRef(null);
  const [menuWidth, setMenuWidth] = useState(undefined);

  const handleMenuOpen = () => {
    if (formControlRef.current) {
      setMenuWidth(formControlRef.current.clientWidth);
    }
  };

  const approvalMenuProps = {
    anchorOrigin: { vertical: "bottom", horizontal: "left" },
    transformOrigin: { vertical: "top", horizontal: "left" },
    PaperProps: {
      style: menuWidth ? { width: menuWidth, maxWidth: menuWidth } : undefined,
    },
  };

  const menuItemSx = {
    fontSize: FIELD_FONT_SIZE,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    display: "block",
  };

  const selectValueSx = {
    "& .MuiSelect-select": {
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap",
    },
  };

  if (interactive) {
    const options = awaiting
      ? APPROVAL_ACTION_OPTIONS
      : UNASSIGNED_ACTION_OPTIONS;
    const placeholder = awaiting ? "Awaiting your approval" : "Unassigned";

    return (
      <FormControl
        ref={formControlRef}
        variant="standard"
        fullWidth
        sx={standardSelectSx}
      >
        <InputLabel shrink>Approval Status</InputLabel>
        <Select
          value={actionValue}
          displayEmpty
          onChange={onActionChange}
          onOpen={handleMenuOpen}
          label="Approval Status"
          IconComponent={ArrowDropDownIcon}
          MenuProps={approvalMenuProps}
          renderValue={(selected) =>
            selected ? (
              selected
            ) : (
              <Box
                component="span"
                sx={{
                  color: TABLE_TEXT_COLOR,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  display: "block",
                }}
              >
                {placeholder}
              </Box>
            )
          }
          sx={selectValueSx}
        >
          {options.map((opt) => (
            <MenuItem key={opt} value={opt} sx={menuItemSx}>
              {opt}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    );
  }

  const displayValue = getApprovalStatusFieldValue(approvalStatus);

  return (
    <StandardSelect
      label="Approval Status"
      value={displayValue}
      options={[displayValue]}
      disabled
      sx={dottedUnderlineSx}
    />
  );
}

function LineItemDetailPanel({
  lineNumber,
  detail,
  detailExpanded,
  approvalAction,
  onApprovalActionChange,
  onApprovalInfoClick,
  onAuditLogClick,
  newDesignEnabled,
  approvalLayoutFinal = true,
  onApprovalLayoutFinalChange,
  onApprove,
  onReject,
  onReassign,
  onUnassign,
}) {
  const miniApprovalBlock = newDesignEnabled ? (
    <MiniApprovalComponent
      key={lineNumber}
      approvalStatus={detail.approvalStatus}
      onApprove={onApprove}
      onReject={onReject}
      onReassign={onReassign}
      onUnassign={onUnassign}
      onCommentsClick={onAuditLogClick}
    />
  ) : null;

  return (
    <>
      <Box
        sx={{
          px: 2,
          py: 1.5,
          display: "flex",
          alignItems: "center",
          gap: 1,
          flexShrink: 0,
        }}
      >
        <Typography sx={{ fontWeight: 400, fontSize: 17, minWidth: 20 }}>
          {lineNumber}
        </Typography>
        <Box sx={{ flex: 1 }} />
        {newDesignEnabled ? (
          <ApprovalLayoutSegmentedToggle
            approvalLayoutFinal={approvalLayoutFinal}
            onChange={onApprovalLayoutFinalChange}
          />
        ) : null}
        <IconButton
          disableRipple
          size="small"
          sx={{
            bgcolor: PURPLE,
            color: "#fff",
            width: 28,
            height: 28,
            ...filledIconFlatSx(PURPLE),
          }}
        >
          <CameraAltIcon sx={{ fontSize: 16 }} />
        </IconButton>
        <Button
          disableRipple
          variant="outlined"
          color="primary"
          size="small"
          sx={detailHeaderSplitButtonSx}
        >
          SPLIT
        </Button>
        <IconButton disableRipple size="small" sx={iconButtonFlatSx}>
          <OpenInFullIcon sx={{ fontSize: 20 }} />
        </IconButton>
        <IconButton disableRipple size="small" sx={iconButtonFlatSx}>
          <MoreVertIcon sx={{ fontSize: 20 }} />
        </IconButton>
      </Box>

      <Box
        sx={{
          px: 2,
          pb: detailExpanded ? 2 : 3.5,
          pt: 1.5,
          overflowY: "scroll",
          overflowX: "hidden",
          flex: 1,
          minHeight: 0,
        }}
      >
        {newDesignEnabled && approvalLayoutFinal ? miniApprovalBlock : null}

        <Box sx={detailFormRowSx}>
          <StandardSelect
            label="Type"
            value={detail.type}
            options={["Deduction", "Invoice Payment", "Credit"]}
          />
          <StandardTextField
            label="Net Amount"
            value={detail.netAmount}
            disabled
          />
        </Box>

        <Box sx={detailFormRowFullSx}>
          <StandardTextField
            label="Invoice Number"
            value={detail.invoiceNumber}
          />
        </Box>

        {newDesignEnabled && !approvalLayoutFinal ? miniApprovalBlock : null}

        {!newDesignEnabled ? (
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr auto 1fr",
              gap: 1.5,
              mb: 2,
              alignItems: "end",
            }}
          >
            <CashAppApprovalStatusSelect
              approvalStatus={detail.approvalStatus}
              actionValue={approvalAction}
              onActionChange={onApprovalActionChange}
            />
            <IconButton
              disableRipple
              size="small"
              onClick={onApprovalInfoClick}
              sx={{
                p: 0,
                flexShrink: 0,
                mb: 0.75,
                "&:hover": { bgcolor: "transparent" },
              }}
            >
              <InfoIcon
                sx={{
                  fontSize: 22,
                  ...infoIconSx,
                }}
              />
            </IconButton>
            <StandardSelect
              label="Deduction Status"
              value={detail.deductionStatus}
              options={["", "Dispute - Filing in progress"]}
              displayEmpty
            />
          </Box>
        ) : null}

        <Box sx={detailFormRowFullSx}>
          <StandardSelect
            label="Clearing"
            value={detail.clearing}
            options={["No", "Yes"]}
          />
        </Box>

        <Box sx={detailFormRowSx}>
          <StandardSelect
            label="Reason"
            value={detail.reason}
            options={["Distributer - Expired at DC"]}
          />
          <StandardSelect
            label="GL Account"
            value={detail.glAccount}
            options={["", "4100-100"]}
            displayEmpty
          />
        </Box>

        <Box sx={detailFormRowFullSx}>
          <StandardSelect
            label="Planning Group"
            value={detail.planningGroup}
            options={["", "Group A"]}
            displayEmpty
          />
        </Box>

        <Box sx={detailFormRowFullSx}>
          <Box sx={{ position: "relative", width: "100%" }}>
            <StandardSelect
              label="Product Family"
              value={detail.productFamily}
              options={["", "Family A"]}
              displayEmpty
              sx={{
                ...rightPanelFieldSx,
                "& .MuiSelect-icon": { right: 28 },
              }}
            />
            <InfoOutlinedIcon
              sx={{
                position: "absolute",
                right: 0,
                bottom: 8,
                fontSize: 18,
                color: ICON_MUTED,
                pointerEvents: "none",
              }}
            />
          </Box>
        </Box>

        <Box sx={detailFormRowFullSx}>
          <StandardSelect
            label="Promo"
            value={detail.promo}
            options={["", "PROMO1"]}
            displayEmpty
          />
        </Box>

        <Box sx={detailFormRowFullSx}>
          <StandardTextField
            label="Description"
            value={detail.description}
            disabled
            sx={dottedUnderlineSx}
          />
        </Box>

        <Box sx={detailFormRowSx}>
          <StandardTextField
            label="Memo"
            value={detail.memo}
            InputProps={{ endAdornment: infoAdornment }}
          />
          <StandardTextField
            label="Execution Start Date"
            value={detail.executionStartDate}
            InputProps={dateInputProps}
          />
        </Box>

        <Box sx={detailFormRowSx}>
          <StandardTextField
            label="Execution End Date"
            value={detail.executionEndDate}
            InputProps={dateInputProps}
          />
          <StandardTextField
            label="Invoice Date"
            value={detail.invoiceDate}
            InputProps={dateInputProps}
          />
        </Box>

        <Box sx={detailFormRowSx}>
          <StandardSelect
            label="DC"
            value={detail.dc}
            options={["", "DC1"]}
            displayEmpty
          />
          <StandardTextField
            label="Rate"
            value={detail.rate}
            InputProps={{ endAdornment: perUnitAdornment }}
          />
        </Box>

        <Box sx={detailFormRowSx}>
          <StandardTextField
            label="Quantity"
            value={detail.quantity}
            InputProps={{ endAdornment: perUnitAdornment }}
          />
          <StandardSelect
            label="Class"
            value={detail.class}
            options={["", "Class A"]}
            displayEmpty
          />
        </Box>
      </Box>
    </>
  );
}

export default function CashAppPage({ newDesignEnabled = false, resetKey = 0 }) {
  const theme = useTheme();
  const [leftTab, setLeftTab] = useState("check");
  const [detailExpanded, setDetailExpanded] = useState(false);
  const [approvalModalOpen, setApprovalModalOpen] = useState(false);
  const [auditLogOpen, setAuditLogOpen] = useState(false);
  const [selectedLineItem, setSelectedLineItem] = useState("1");
  const [lineItemDetails, setLineItemDetails] = useState(() =>
    cloneDefaultLineItemDetails(),
  );
  const [approvalAction, setApprovalAction] = useState("");
  const [approvalLayoutFinal, setApprovalLayoutFinal] = useState(true);

  useEffect(() => {
    setApprovalAction("");
    setAuditLogOpen(false);
  }, [selectedLineItem]);

  const selectedDetail =
    lineItemDetails[selectedLineItem] ?? lineItemDetails["1"];

  const navigableLineIndex = NAVIGABLE_LINE_ITEMS.indexOf(selectedLineItem);
  const activeNavigableIndex =
    navigableLineIndex >= 0 ? navigableLineIndex : 0;
  const canGoBackLineItem = activeNavigableIndex > 0;
  const canGoNextLineItem =
    activeNavigableIndex < NAVIGABLE_LINE_ITEMS.length - 1;
  const lineItemProgress =
    ((activeNavigableIndex + 1) / NAVIGABLE_LINE_ITEMS.length) * 100;

  const handleBackLineItem = () => {
    if (!canGoBackLineItem) return;
    setSelectedLineItem(NAVIGABLE_LINE_ITEMS[activeNavigableIndex - 1]);
  };

  const handleNextLineItem = () => {
    if (!canGoNextLineItem) return;
    setSelectedLineItem(NAVIGABLE_LINE_ITEMS[activeNavigableIndex + 1]);
  };

  const applyApprovalToSelectedLineItem = (action, assignee) => {
    setLineItemDetails((prev) => {
      const current = prev[selectedLineItem];
      if (!current) return prev;
      return {
        ...prev,
        [selectedLineItem]: applyLineItemApprovalAction(
          current,
          action,
          assignee,
        ),
      };
    });
    setApprovalAction("");
  };

  const applyManageActionToSelectedLineItem = (applyFn, payload) => {
    setLineItemDetails((prev) => {
      const current = prev[selectedLineItem];
      if (!current) return prev;
      return {
        ...prev,
        [selectedLineItem]: applyFn(current, payload),
      };
    });
  };

  const handleRejectManage = (payload) => {
    applyManageActionToSelectedLineItem(applyRejectManageAction, payload);
  };

  const handleReassignManage = (payload) => {
    applyManageActionToSelectedLineItem(applyReassignManageAction, payload);
  };

  const handleUnassignManage = (payload) => {
    applyManageActionToSelectedLineItem(applyUnassignManageAction, payload);
  };

  const handleReassign = ({ to }) => {
    const assignee = to[0];
    if (!assignee) return;
    applyReassignToSelectedLineItem(assignee);
  };

  const applyReassignToSelectedLineItem = (assignee) => {
    if (!assignee) return;
    setLineItemDetails((prev) => {
      const current = prev[selectedLineItem];
      if (!current) return prev;
      return {
        ...prev,
        [selectedLineItem]: applyLineItemApprovalAction(
          current,
          assignee === "Unassign" ? "unassign" : "assign",
          assignee === "Unassign" ? null : assignee,
        ),
      };
    });
  };

  const handleApprovalActionChange = (event) => {
    const action = event.target.value;
    if (action === "Approve") {
      applyApprovalToSelectedLineItem("approve");
      return;
    }
    if (action === "Reject") {
      applyApprovalToSelectedLineItem("reject");
      return;
    }
    if (
      UNASSIGNED_ASSIGNEE_OPTIONS.includes(action) &&
      isUnassignedStatus(selectedDetail.approvalStatus)
    ) {
      applyApprovalToSelectedLineItem("assign", action);
      return;
    }
    setApprovalAction(action);
  };

  const handleResetCashApp = () => {
    setLeftTab("check");
    setDetailExpanded(false);
    setApprovalModalOpen(false);
    setAuditLogOpen(false);
    setSelectedLineItem("1");
    setLineItemDetails(cloneDefaultLineItemDetails());
    setApprovalAction("");
    setApprovalLayoutFinal(true);
  };

  useEffect(() => {
    if (resetKey === 0) return;
    handleResetCashApp();
  }, [resetKey]);

  const infoBoxSx = {
    p: 1.5,
    borderRadius: 1,
    bgcolor: alpha(theme.palette.primary.main, 0.08),
    border: `1px solid ${alpha(theme.palette.primary.main, 0.25)}`,
    display: "flex",
    alignItems: "flex-start",
    gap: 1,
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100%",
        overflow: "auto",
        bgcolor: "#f5f5f5",
        pb: 2,
      }}
    >
      {/* Page header */}
      <Box
        sx={{
          mx: 2,
          mt: 2,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 2,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography
            sx={{ fontSize: 15, color: HEADER_TEXT_COLOR, fontWeight: 400 }}
          >
            CSV Upload Payment Remittance
          </Typography>
          <IconButton
            disableRipple
            size="small"
            sx={{ color: ICON_MUTED, p: 0.5, ...iconButtonFlatSx }}
          >
            <DeleteIcon sx={{ fontSize: 21 }} />
          </IconButton>
          <IconButton
            disableRipple
            size="small"
            sx={{ color: ICON_MUTED, p: 0.5, ...iconButtonFlatSx }}
          >
            <GetAppIcon sx={{ fontSize: 21 }} />
          </IconButton>
        </Box>
      </Box>

      {/* Check header card */}
      <Box
        sx={{
          bgcolor: "#fff",
          mx: 2,
          mt: 1.5,
          borderRadius: 1.5,
          border: "1px solid #e8e8e8",
          boxShadow: "0 1px 3px rgba(0, 0, 0, 0.06)",
          overflow: "visible",
        }}
      >
        <Box
          sx={{
            px: 3,
            pt: 2,
            pb: 1.5,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Typography sx={{ fontWeight: 400, fontSize: 19, color: "#111827" }}>
            Check #1214470
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <IconButton
              disableRipple
              size="small"
              sx={{ color: ICON_MUTED, ...iconButtonFlatSx }}
            >
              <ArrowDropUpIcon sx={buttonChevronLgSx} />
            </IconButton>
            <Button
              disableRipple
              variant="contained"
              size="small"
              color="primary"
              sx={{
                fontSize: 13,
                fontWeight: 400,
                px: 2,
                color: "#fff",
                ...containedFlatSx,
              }}
            >
              UNAPPLIED
            </Button>
          </Box>
        </Box>

        <Box sx={{ display: "flex", alignItems: "stretch" }}>
          <Box sx={{ flex: 1, minWidth: 0, px: 3, pb: 2.5, pt: 0.5 }}>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                gap: 2,
                mb: 2,
              }}
            >
              <StandardSelect
                label="Customer Name *"
                value="KeHE"
                options={["KeHE", "UNFI East", "UNFI West"]}
              />
              <StandardTextField label="Check Total Amount" value="$7,910.08" />
              <StandardTextField
                label="Deduction Amount"
                value="-$40,630.39"
                disabled
              />
            </Box>

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                gap: 2,
              }}
            >
              <StandardTextField
                label="Check Date"
                value="07/04/2025"
                InputProps={dateInputProps}
                sx={dateFieldSx}
              />
              <StandardTextField
                label="Deposit Date"
                value=""
                InputProps={dateInputProps}
                sx={dateFieldSx}
              />
              <StandardTextField label="Check Number" value="1214470" />
              <StandardSelect
                label="Deposit Account"
                value=""
                options={["", "Unapplied Cash Payable"]}
                displayEmpty
              />
            </Box>
          </Box>

          <Box
            sx={{
              width: 300,
              flexShrink: 0,
              borderLeft: "1px solid #ebebeb",
              px: 2.5,
              py: 0.5,
              pb: 2.5,
              display: "flex",
              flexDirection: "column",
              gap: 1.5,
            }}
          >
            <Box sx={infoBoxSx}>
              <InfoOutlinedIcon
                sx={{
                  fontSize: 17,
                  color: "primary.main",
                  mt: 0.1,
                  flexShrink: 0,
                }}
              />
              <Typography
                sx={{
                  fontSize: 13,
                  color: "#374151",
                  flex: 1,
                  lineHeight: 1.5,
                }}
              >
                This payment has not been validated, connect your bank.
              </Typography>
              <Button
                disableRipple
                size="small"
                sx={{
                  fontSize: 12,
                  fontWeight: 400,
                  color: "primary.main",
                  minWidth: 0,
                  p: 0,
                  alignSelf: "center",
                  flexShrink: 0,
                  ...textButtonFlatSx,
                }}
              >
                CONNECT
              </Button>
            </Box>
            <Button
              disableRipple
              variant="outlined"
              size="small"
              fullWidth
              startIcon={<RefreshIcon sx={{ fontSize: 15 }} />}
              sx={{
                fontSize: 13,
                fontWeight: 400,
                borderColor: PURPLE,
                color: PURPLE,
                whiteSpace: "nowrap",
                ...outlinedFlatSx(PURPLE, PURPLE),
              }}
            >
              REFRESH CHECK INVOICES
            </Button>
          </Box>
        </Box>
      </Box>

      {/* Main content — left table + right detail/line items */}
      <Box
        sx={{
          display: "flex",
          gap: 2,
          mx: 2,
          mt: 3.5,
          alignItems: "flex-start",
        }}
      >
        {/* Left column */}
        <Box
          sx={{
            flex: "1 1 0",
            minWidth: 0,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <Box
            sx={{
              display: "flex",
              alignItems: "flex-end",
              gap: 4,
              mb: "-1px",
              position: "relative",
              zIndex: 1,
            }}
          >
            {[
              { id: "check", label: "CHECK" },
              { id: "invoice", label: "INVOICE" },
            ].map((tab) => (
              <Box
                key={tab.id}
                onClick={() => setLeftTab(tab.id)}
                sx={{
                  borderBottom: "2px solid",
                  borderColor:
                    leftTab === tab.id ? "primary.main" : "transparent",
                  pb: 1,
                  px: 2.5,
                  minWidth: 88,
                  cursor: "pointer",
                }}
              >
                <Typography
                  sx={{
                    fontSize: 15,
                    fontWeight: 300,
                    color: leftTab === tab.id ? "primary.main" : ICON_MUTED,
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                  }}
                >
                  {tab.label}
                </Typography>
              </Box>
            ))}
          </Box>

          <Box
            sx={{
              bgcolor: "#fff",
              borderRadius: 1.5,
              border: "1px solid #e0e0e0",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              minHeight: 500,
              maxHeight: 560,
            }}
          >
            {leftTab === "check" ? (
              <>
                <Box
                  sx={{
                    px: 2,
                    py: 1,
                    display: "flex",
                    alignItems: "center",
                    gap: 2.5,
                    flexWrap: "wrap",
                    borderBottom: "1px solid #f0f0f0",
                    flexShrink: 0,
                  }}
                >
                  {LEFT_TOOLBAR_ITEMS.map(({ label, icon: Icon }) => (
                    <Button
                      key={label}
                      disableRipple
                      size="small"
                      startIcon={
                        <Icon sx={{ fontSize: 16, color: "primary.main" }} />
                      }
                      sx={{
                        fontSize: 12,
                        fontWeight: 400,
                        color: "primary.main",
                        minWidth: 0,
                        p: 0,
                        ...textButtonFlatSx,
                      }}
                    >
                      {label}
                    </Button>
                  ))}
                </Box>

                <TableContainer sx={{ flex: 1, overflow: "auto" }}>
                  <Table
                    size="small"
                    stickyHeader
                    sx={{
                      tableLayout: "fixed",
                      minWidth: CHECK_TABLE_MIN_WIDTH,
                      width: "max-content",
                    }}
                  >
                    <TableHead
                      sx={{ "& .MuiTableCell-head": { bgcolor: "#fff" } }}
                    >
                      <TableRow>
                        {CHECK_TABLE_COLUMNS.map((col) => (
                          <TableCell
                            key={col.id}
                            sx={{
                              ...tableHeadCellSx,
                              minWidth: col.minWidth,
                              width: col.minWidth,
                            }}
                          >
                            {col.label}
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {CHECK_TABLE_ROWS.map((row) => (
                        <TableRow
                          key={row.invoiceNumber}
                          sx={{
                            bgcolor: row.highlight
                              ? alpha(theme.palette.primary.main, 0.12)
                              : "transparent",
                          }}
                        >
                          {CHECK_TABLE_COLUMNS.map((col) => (
                            <TableCell
                              key={col.id}
                              sx={{
                                ...tableBodyCellSx,
                                minWidth: col.minWidth,
                                width: col.minWidth,
                              }}
                            >
                              {row[col.id]}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </>
            ) : (
              <>
                <Box
                  sx={{
                    px: 2,
                    py: 1.25,
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    flexShrink: 0,
                    borderBottom: "1px solid #f0f0f0",
                  }}
                >
                  <IconButton
                    disableRipple
                    size="small"
                    sx={docToolIconButtonSx}
                  >
                    <RemoveIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                  <IconButton
                    disableRipple
                    size="small"
                    sx={docToolIconButtonSx}
                  >
                    <AddIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                  <IconButton
                    disableRipple
                    size="small"
                    sx={docToolIconButtonSx}
                  >
                    <RefreshIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                  <Box sx={{ flex: 1 }} />
                  <IconButton disableRipple size="small" sx={iconButtonFlatSx}>
                    <OpenInFullIcon sx={{ fontSize: 20 }} />
                  </IconButton>
                </Box>
                <Box
                  sx={{
                    flex: 1,
                    minHeight: 440,
                    bgcolor: "#8a8a8a",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Button
                    disableRipple
                    variant="contained"
                    color="primary"
                    sx={{
                      fontSize: 14,
                      fontWeight: 600,
                      px: 3.5,
                      py: 1.1,
                      borderRadius: 1.5,
                      textTransform: "none",
                      color: "#fff",
                      ...containedFlatSx,
                    }}
                  >
                    Upload Remittance Document
                  </Button>
                </Box>
              </>
            )}
          </Box>

          <Box
            sx={{
              mt: 2,
              bgcolor: "#fff",
              borderRadius: 1.5,
              border: "1px solid #e0e0e0",
              px: 3,
              py: 3.5,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 3,
            }}
          >
            <Box>
              <Typography
                sx={{
                  fontSize: 17,
                  color: "primary.main",
                  mb: 1,
                  lineHeight: 1.4,
                }}
              >
                Check Net Amount:{" "}
                <Box
                  component="span"
                  sx={{ color: "#111827", fontWeight: 300, fontSize: 20 }}
                >
                  $19,608.58
                </Box>
              </Typography>
              <Typography
                sx={{ fontSize: 17, color: "primary.main", lineHeight: 1.4 }}
              >
                Calculated Net Amount:{" "}
                <Box
                  component="span"
                  sx={{ color: "#111827", fontWeight: 300, fontSize: 20 }}
                >
                  $19,608.58
                </Box>
              </Typography>
            </Box>
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: 1.5,
                minWidth: 190,
              }}
            >
              <Button
                disableRipple
                variant="contained"
                color="primary"
                size="medium"
                fullWidth
                sx={{
                  fontSize: 14,
                  fontWeight: 400,
                  color: "#fff",
                  py: 0.75,
                  ...containedFlatSx,
                }}
              >
                SAVE
              </Button>
              <Button
                disableRipple
                variant="outlined"
                color="primary"
                size="medium"
                fullWidth
                sx={{
                  fontSize: 14,
                  fontWeight: 400,
                  py: 0.75,
                  ...outlinedFlatSx("primary.main", "primary.main", "#fff"),
                }}
              >
                READY FOR REVIEW
              </Button>
            </Box>
          </Box>
        </Box>

        {/* Right column */}
        <Box
          sx={{
            flex: "1 1 0",
            minWidth: 0,
            display: "flex",
            flexDirection: "column",
            gap: 2,
            ...rightColumnInputSx,
          }}
        >
          {/* Detail panel */}
          <Box
            sx={{
              position: "relative",
              flexShrink: 0,
              zIndex: 1,
              mb: detailExpanded ? 0 : 2.5,
            }}
          >
            <Box
              sx={{
                bgcolor: "#fff",
                borderRadius: 1.5,
                border: "1px solid #e0e0e0",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                maxHeight: detailExpanded
                  ? newDesignEnabled
                    ? DETAIL_PANEL_MAX_HEIGHT.expanded.newDesign
                    : DETAIL_PANEL_MAX_HEIGHT.expanded.default
                  : newDesignEnabled
                    ? DETAIL_PANEL_MAX_HEIGHT.collapsed.newDesign
                    : DETAIL_PANEL_MAX_HEIGHT.collapsed.default,
              }}
            >
              <LineItemDetailPanel
                lineNumber={selectedLineItem}
                detail={selectedDetail}
                detailExpanded={detailExpanded}
                approvalAction={approvalAction}
                onApprovalActionChange={handleApprovalActionChange}
                onApprovalInfoClick={() => setApprovalModalOpen(true)}
                onAuditLogClick={() => setAuditLogOpen(true)}
                newDesignEnabled={newDesignEnabled}
                approvalLayoutFinal={approvalLayoutFinal}
                onApprovalLayoutFinalChange={setApprovalLayoutFinal}
                onApprove={() => applyApprovalToSelectedLineItem("approve")}
                onReject={handleRejectManage}
                onReassign={handleReassignManage}
                onUnassign={handleUnassignManage}
              />
            </Box>

            {!detailExpanded ? (
              <DetailExpandToggle onClick={() => setDetailExpanded(true)} />
            ) : null}
          </Box>

          {/* Line items panel */}
          <Box
            sx={{ display: "flex", flexDirection: "column", flex: 1, mt: 2 }}
          >
            <Box
              sx={{
                px: 2,
                pt: 2,
                pb: 1.25,
                display: "flex",
                alignItems: "center",
                gap: 2,
                mb: "-1px",
                position: "relative",
                zIndex: 1,
              }}
            >
              <Button
                disableRipple
                variant="outlined"
                size="small"
                disabled={!canGoBackLineItem}
                onClick={handleBackLineItem}
                sx={
                  canGoBackLineItem
                    ? lineNavActiveButtonSx
                    : lineNavDisabledButtonSx
                }
              >
                BACK
              </Button>
              <Box sx={{ flex: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={lineItemProgress}
                  sx={{
                    height: 4,
                    borderRadius: 2,
                    bgcolor: "#e8e8e8",
                    "& .MuiLinearProgress-bar": {
                      bgcolor: alpha(theme.palette.primary.main, 0.28),
                    },
                  }}
                />
              </Box>
              <Button
                disableRipple
                variant="outlined"
                color="primary"
                size="small"
                disabled={!canGoNextLineItem}
                onClick={handleNextLineItem}
                sx={
                  canGoNextLineItem
                    ? lineNavActiveButtonSx
                    : lineNavDisabledButtonSx
                }
              >
                NEXT
              </Button>
            </Box>

            <Box
              sx={{
                bgcolor: "#fff",
                borderRadius: 1.5,
                border: "1px solid #e0e0e0",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                flex: 1,
              }}
            >
              <Box
                sx={{
                  px: 2,
                  py: 1.5,
                  display: "flex",
                  alignItems: "center",
                  gap: 1.5,
                  flexWrap: "wrap",
                  borderBottom: "1px solid #f0f0f0",
                }}
              >
                <SearchIcon sx={{ fontSize: 20, color: ICON_MUTED }} />
                <TextField
                  placeholder="Invoice Number"
                  variant="standard"
                  sx={{
                    width: 150,
                    ...standardFieldSx,
                  }}
                />
                <FormControl
                  variant="outlined"
                  size="small"
                  sx={{
                    minWidth: 88,
                    "& .MuiInputLabel-root": {
                      fontSize: RIGHT_LABEL_FONT_SIZE,
                      color: LABEL_COLOR,
                    },
                    "& .MuiOutlinedInput-root": {
                      fontSize: RIGHT_FIELD_FONT_SIZE,
                      color: "#111827",
                      "& .MuiOutlinedInput-notchedOutline": {
                        borderColor: "#d1d5db",
                      },
                      "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                        borderColor: "primary.main",
                      },
                    },
                    "& .MuiSelect-icon": { color: ICON_MUTED, fontSize: 20 },
                  }}
                >
                  <InputLabel>Type</InputLabel>
                  <Select
                    label="Type"
                    value="All"
                    IconComponent={ArrowDropDownIcon}
                  >
                    <MenuItem
                      value="All"
                      sx={{ fontSize: RIGHT_FIELD_FONT_SIZE }}
                    >
                      All
                    </MenuItem>
                  </Select>
                </FormControl>
                <Box sx={{ flex: 1, minWidth: 16 }} />
                <Button
                  disableRipple
                  variant="contained"
                  color="primary"
                  size="small"
                  sx={{
                    fontSize: 13,
                    fontWeight: 400,
                    color: "#fff",
                    ...containedFlatSx,
                  }}
                >
                  EXPAND
                </Button>
                <Button
                  disableRipple
                  variant="contained"
                  color="primary"
                  size="small"
                  sx={{
                    fontSize: 13,
                    fontWeight: 400,
                    color: "#fff",
                    ...containedFlatSx,
                  }}
                >
                  BULK UPDATE
                </Button>
                <Button
                  disableRipple
                  variant="outlined"
                  color="primary"
                  size="small"
                  sx={{
                    fontSize: 13,
                    fontWeight: 400,
                    ...outlinedFlatSx("primary.main", "primary.main", "#fff"),
                  }}
                >
                  ADD LINE
                </Button>
              </Box>

              <TableContainer sx={{ maxHeight: 260 }}>
                <Table
                  size="small"
                  sx={{ tableLayout: "fixed", width: "100%" }}
                >
                  <TableHead
                    sx={{ "& .MuiTableCell-head": { bgcolor: "#fff" } }}
                  >
                    <TableRow>
                      {[
                        "Line Item",
                        "Invoice Number",
                        "Amount",
                        "Type",
                        "View",
                      ].map((col) => (
                        <TableCell key={col} sx={rightTableHeadCellSx}>
                          {col}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {LINE_ITEM_ROWS.map((row) => {
                      const isSelected = selectedLineItem === row.line;
                      const hasDetail = Boolean(LINE_ITEM_DETAILS[row.line]);

                      return (
                        <TableRow
                          key={row.line}
                          onClick={
                            hasDetail
                              ? () => setSelectedLineItem(row.line)
                              : undefined
                          }
                          sx={{
                            bgcolor: isSelected
                              ? alpha(theme.palette.primary.main, 0.12)
                              : "transparent",
                            cursor: hasDetail ? "pointer" : "default",
                            "&:hover": hasDetail
                              ? {
                                  bgcolor: alpha(
                                    theme.palette.primary.main,
                                    isSelected ? 0.12 : 0.06,
                                  ),
                                }
                              : undefined,
                          }}
                        >
                          <TableCell sx={rightTableBodyCellSx}>
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 0.5,
                              }}
                            >
                              {row.expandable && (
                                <KeyboardArrowRightIcon
                                  sx={{ fontSize: 17, color: ICON_MUTED }}
                                />
                              )}
                              {row.line}
                            </Box>
                          </TableCell>
                          <TableCell sx={rightTableBodyCellSx}>
                            {row.invoiceNumber}
                          </TableCell>
                          <TableCell sx={rightTableBodyCellSx}>
                            {row.amount}
                          </TableCell>
                          <TableCell sx={rightTableBodyCellSx}>
                            {row.type}
                          </TableCell>
                          <TableCell
                            sx={{ ...rightTableBodyCellSx, width: 48 }}
                          >
                            <IconButton
                              disableRipple
                              size="small"
                              sx={{ p: 0.25, ...iconButtonFlatSx }}
                            >
                              <OpenInNewIcon
                                sx={{ fontSize: 21, color: ICON_MUTED }}
                              />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </Box>
        </Box>
      </Box>

      <ViewAuditLogPopover
        open={auditLogOpen}
        onClose={() => setAuditLogOpen(false)}
        approvalHistory={selectedDetail.approvalHistory}
      />

      <ViewApprovalModal
        open={approvalModalOpen}
        onClose={() => setApprovalModalOpen(false)}
        approvalStatus={selectedDetail.approvalStatus}
        approvalHistory={selectedDetail.approvalHistory}
        canApprove={isAwaitingCurrentUserApproval(
          selectedDetail.approvalStatus,
        )}
        onApprove={() => applyApprovalToSelectedLineItem("approve")}
        onReject={() => applyApprovalToSelectedLineItem("reject")}
        onReassign={handleReassign}
      />
    </Box>
  );
}
