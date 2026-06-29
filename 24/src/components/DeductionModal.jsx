import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import {
  ArrowDropDownIcon,
  KeyboardArrowUpIcon,
  buttonChevronSx,
} from "../theme/icons";
import KeyboardArrowRightIcon from "@mui/icons-material/KeyboardArrowRight";
import UploadIcon from "@mui/icons-material/Upload";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import FilterListIcon from "@mui/icons-material/FilterList";
import DensityMediumIcon from "@mui/icons-material/DensityMedium";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import { alpha, useTheme } from "@mui/material/styles";
import { useEffect, useState } from "react";
import ApprovalHistoryStepper from "./ApprovalHistoryStepper";
import DeductionModalApprovalsAudit from "./DeductionModalApprovalsAudit";
import ModalCommentsSection from "./ModalCommentsSection";
import IosToggle from "./IosToggle";
import {
  TABLE_TEXT_COLOR,
  AWAITING_APPROVAL_ORANGE,
  PENDING_APPROVAL_BLUE,
} from "../theme/colors";

const GREEN = "#2e7d32";
const REJECT_RED = "#d32f2f";
const LABEL_COLOR = "#6b7280";

const detailAccordionCardSx = {
  border: "1px solid #e8eaed",
  borderRadius: 1,
  bgcolor: "#fff",
  boxShadow: "4px 4px 8px -4px rgba(0, 0, 0, 0.12)",
  mb: 1.5,
  overflow: "visible",
};

const panelTitleSx = {
  fontSize: 15,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
};

const detailAccordionHeaderSx = {
  px: 2,
  py: 1.5,
  display: "flex",
  alignItems: "center",
  cursor: "pointer",
};

const accordionContentIndentPx = 16 + 20 + 10;

const accordionBodyScrollSx = {
  overflowX: "hidden",
};

const visibleVerticalScrollbarSx = {
  overflowY: "scroll",
  overflowX: "hidden",
  scrollbarWidth: "auto",
  scrollbarColor: "#bdbdbd #f0f0f0",
  "&::-webkit-scrollbar": {
    width: 12,
  },
  "&::-webkit-scrollbar-track": {
    backgroundColor: "#f0f0f0",
    borderRadius: 6,
  },
  "&::-webkit-scrollbar-thumb": {
    backgroundColor: "#bdbdbd",
    borderRadius: 6,
    border: "2px solid #f0f0f0",
  },
  "&::-webkit-scrollbar-thumb:hover": {
    backgroundColor: "#9e9e9e",
  },
};

const tableScrollSx = {
  overflowX: "scroll",
  overflowY: "hidden",
  scrollbarWidth: "auto",
  scrollbarColor: "#bdbdbd #f0f0f0",
  "&::-webkit-scrollbar": {
    height: 12,
  },
  "&::-webkit-scrollbar-track": {
    backgroundColor: "#f0f0f0",
    borderRadius: 6,
  },
  "&::-webkit-scrollbar-thumb": {
    backgroundColor: "#bdbdbd",
    borderRadius: 6,
    border: "2px solid #f0f0f0",
  },
  "&::-webkit-scrollbar-thumb:hover": {
    backgroundColor: "#9e9e9e",
  },
};

const approvalPanelScrollSx = {
  maxHeight: 320,
  ...visibleVerticalScrollbarSx,
};

const accordionSectionLabelSx = {
  fontSize: 14,
  fontWeight: 400,
  color: LABEL_COLOR,
  lineHeight: 1.4,
};

const textButtonFlatSx = {
  fontWeight: 400,
  "&:hover": { bgcolor: "transparent" },
  "&:active": { bgcolor: "transparent" },
};

const flatContainedSx = {
  boxShadow: "none",
  fontWeight: 400,
  "&:hover": { boxShadow: "none", bgcolor: "primary.main" },
  "&:active": { boxShadow: "none", bgcolor: "primary.main" },
};

const flatOutlinedSx = (borderColor, color) => ({
  boxShadow: "none",
  fontWeight: 400,
  "&:hover": { borderColor, color, bgcolor: "transparent", boxShadow: "none" },
  "&:active": { borderColor, color, bgcolor: "transparent", boxShadow: "none" },
});

const modalButtonHeightSx = {
  minHeight: 32,
  py: 0.625,
};

const modalResetButtonSx = {
  fontSize: 14,
  fontWeight: 400,
  p: 0,
  minWidth: 0,
  minHeight: 0,
  lineHeight: 1,
  color: "primary.main",
  "&:hover": { bgcolor: "transparent" },
};

const modalRejectButtonSx = {
  fontSize: 12,
  fontWeight: 400,
  px: 1.5,
  minWidth: 0,
  flexShrink: 0,
  color: REJECT_RED,
  borderColor: REJECT_RED,
  ...modalButtonHeightSx,
  ...flatOutlinedSx(REJECT_RED, REJECT_RED),
};

const modalApproveButtonSx = {
  fontSize: 12,
  fontWeight: 400,
  px: 1.5,
  minWidth: 0,
  flexShrink: 0,
  color: "#fff",
  bgcolor: GREEN,
  borderColor: GREEN,
  ...modalButtonHeightSx,
  boxShadow: "none",
  "&:hover": {
    bgcolor: GREEN,
    borderColor: GREEN,
    boxShadow: "none",
  },
  "&:active": {
    bgcolor: GREEN,
    borderColor: GREEN,
    boxShadow: "none",
  },
};

const reassignActionButtonSx = {
  fontSize: 12,
  fontWeight: 400,
  px: 1.5,
  minWidth: 0,
  flexShrink: 0,
  ...modalButtonHeightSx,
  ...flatOutlinedSx("primary.main", "primary.main"),
};

const reassignSelectFormSx = {
  "& .MuiOutlinedInput-root": {
    fontSize: 14,
    fontWeight: 400,
    color: TABLE_TEXT_COLOR,
    bgcolor: "#fff",
    minHeight: 40,
    py: 0.75,
    "& fieldset": {
      borderColor: "primary.main",
    },
    "&:hover fieldset": {
      borderColor: "primary.main",
    },
    "&.Mui-focused fieldset": {
      borderColor: "primary.main",
      borderWidth: 1,
    },
  },
  "& .MuiSelect-select": {
    py: 0,
    px: 1.25,
    display: "flex",
    alignItems: "center",
  },
  "& .MuiSelect-icon": {
    color: LABEL_COLOR,
    fontSize: 20,
    right: 8,
  },
};

const reassignMenuItemSx = {
  fontSize: 14,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
};

const REASSIGN_ASSIGNEE_OPTIONS = [
  "Unassign",
  "Adrian Cardenas",
  "Adrien",
  "Allys",
  "Ben",
  "ben1",
  "Beverly",
  "Broker1",
  "Kevin",
  "Odette",
  "Matt",
  "Justin Hunter",
  "kevexternal",
];

const tableBodyCellSx = {
  fontSize: 14,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  borderBottom: "1px solid #f0f0f0",
  py: 1.75,
  px: 1.5,
  whiteSpace: "nowrap",
};

const tableHeadCellSx = {
  ...tableBodyCellSx,
  borderBottom: "1px solid #e0e0e0",
  bgcolor: "#fff",
};

const CHECK_ROW_TEMPLATE = {
  invoiceDate: "6/3/2023",
  invoiceNbr: "CM060323026141CAN1",
  poNbr: "8391993",
  discountAmt: "0.00",
  invoiceAmt: "1,850.03",
  checkAmt: "-1,850.03",
  voucherNbr: "70001326861",
};

const CHECK_ROWS = Array.from({ length: 8 }, (_, i) => ({
  ...CHECK_ROW_TEMPLATE,
  voucherNbr: `700013268${String(61 + i).padStart(2, "0")}`,
  highlight: i === 0,
}));

const TOOLBAR_ITEMS = [
  { label: "COLUMNS", icon: ViewColumnIcon },
  { label: "FILTERS", icon: FilterListIcon },
  { label: "DENSITY", icon: DensityMediumIcon },
  { label: "EXPORT", icon: FileDownloadIcon },
];

const CURRENT_USER = "Beverly";

function getAssigneeName(approvalStatus) {
  if (!approvalStatus?.startsWith("Assigned to ")) return null;
  return approvalStatus.slice("Assigned to ".length);
}

function isAwaitingCurrentUserApproval(approvalStatus) {
  return getAssigneeName(approvalStatus) === CURRENT_USER;
}

function isUnassignedStatus(approvalStatus) {
  return !approvalStatus || approvalStatus === "Unassigned";
}

function isApprovedStatus(approvalStatus) {
  return approvalStatus === "Approved";
}

function isRejectedStatus(approvalStatus) {
  return approvalStatus === "Rejected";
}

function isResolvedApprovalStatus(approvalStatus) {
  return isApprovedStatus(approvalStatus) || isRejectedStatus(approvalStatus);
}

function getApprovalChipLabel(approvalStatus) {
  if (isAwaitingCurrentUserApproval(approvalStatus)) {
    return "Awaiting Your Approval";
  }
  if (isApprovedStatus(approvalStatus)) {
    return "Approved";
  }
  if (isRejectedStatus(approvalStatus)) {
    return "Rejected";
  }
  if (isUnassignedStatus(approvalStatus)) {
    return "Unassigned";
  }
  return "Pending Approval";
}

function getApprovalChipSx(approvalStatus) {
  if (isAwaitingCurrentUserApproval(approvalStatus)) {
    return { bgcolor: AWAITING_APPROVAL_ORANGE, color: "#fff" };
  }
  if (isApprovedStatus(approvalStatus)) {
    return { bgcolor: GREEN, color: "#fff" };
  }
  if (isRejectedStatus(approvalStatus)) {
    return { bgcolor: REJECT_RED, color: "#fff" };
  }
  if (isUnassignedStatus(approvalStatus)) {
    return { bgcolor: "#e5e7eb", color: TABLE_TEXT_COLOR };
  }
  return { bgcolor: PENDING_APPROVAL_BLUE, color: "#fff" };
}

export default function DeductionModal({
  open,
  onClose,
  newDesignEnabled = false,
  approvalStatus,
  approvalHistory,
  onApprove,
  onReject,
  onReassign,
  onUnassign,
  onReset,
}) {
  const theme = useTheme();
  const [approvalOpen, setApprovalOpen] = useState(false);
  const [modalNewDesignEnabled, setModalNewDesignEnabled] = useState(false);
  const [isReassigning, setIsReassigning] = useState(false);
  const [reassignMenuOpen, setReassignMenuOpen] = useState(false);
  const [pendingAssignee, setPendingAssignee] = useState("");
  const [displayAssignee, setDisplayAssignee] = useState("");
  const awaitingCurrentUserApproval =
    isAwaitingCurrentUserApproval(approvalStatus);
  const assigneeName = getAssigneeName(approvalStatus);
  const unassigned = isUnassignedStatus(approvalStatus);
  const resolvedApproval = isResolvedApprovalStatus(approvalStatus);
  const showAwaitingUserUi = awaitingCurrentUserApproval;
  const showAssignButton = unassigned || resolvedApproval;
  const approvalChipLabel = getApprovalChipLabel(approvalStatus);
  const approvalChipSx = getApprovalChipSx(approvalStatus);
  const assignedToDisplay =
    unassigned || resolvedApproval
      ? "Unassigned"
      : showAwaitingUserUi
        ? displayAssignee || assigneeName || CURRENT_USER
        : assigneeName || displayAssignee || "Unassigned";

  useEffect(() => {
    if (!open) {
      setIsReassigning(false);
      setReassignMenuOpen(false);
      setPendingAssignee("");
    }
  }, [open]);

  useEffect(() => {
    if (open) {
      setModalNewDesignEnabled(newDesignEnabled);
    }
  }, [open, newDesignEnabled]);

  useEffect(() => {
    setDisplayAssignee(assigneeName || "");
    setIsReassigning(false);
    setReassignMenuOpen(false);
    setPendingAssignee("");
  }, [approvalStatus, open, assigneeName]);

  const handleOpenAssigneeEdit = () => {
    setPendingAssignee(
      showAssignButton ? "" : displayAssignee || assigneeName || CURRENT_USER,
    );
    setIsReassigning(true);
    setReassignMenuOpen(true);
  };

  const handleAssigneeDone = () => {
    if (!pendingAssignee) return;
    setIsReassigning(false);
    setReassignMenuOpen(false);
    setDisplayAssignee(pendingAssignee);
    onReassign?.({ assignee: pendingAssignee });
  };

  const resetModalUiState = () => {
    setModalNewDesignEnabled(newDesignEnabled);
    setApprovalOpen(false);
    setIsReassigning(false);
    setReassignMenuOpen(false);
    setPendingAssignee("");
  };

  const handleModalReset = () => {
    resetModalUiState();
    onReset?.();
  };

  const reassignMenuProps = {
    PaperProps: {
      sx: {
        maxHeight: 280,
        zIndex: theme.zIndex.modal + 2,
      },
    },
    anchorOrigin: { vertical: "bottom", horizontal: "left" },
    transformOrigin: { vertical: "top", horizontal: "left" },
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth={false}
      PaperProps={{
        sx: {
          borderRadius: 1,
          maxHeight: "86vh",
          width: "min(94vw, 1480px)",
          maxWidth: "94vw",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        },
      }}
    >
      <DialogContent
        sx={{
          px: 4,
          pt: 4,
          pb: 2,
          bgcolor: "#f5f5f5",
          overflow: "hidden",
          flex: 1,
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1.5,
            flexWrap: "wrap",
            mb: 2.5,
            flexShrink: 0,
          }}
        >
          <Typography sx={{ fontWeight: 300, fontSize: 20, color: "#111827" }}>
            Invoice: 1326861
          </Typography>
          <Chip
            label="In Review"
            size="small"
            sx={{
              fontSize: 13,
              fontWeight: 400,
              height: 30,
              px: 0.5,
              bgcolor: "#e5e7eb",
              color: TABLE_TEXT_COLOR,
              borderRadius: "999px",
            }}
          />
          <Chip
            label={approvalChipLabel}
            size="small"
            sx={{
              fontSize: 13,
              fontWeight: 400,
              height: 30,
              px: 0.5,
              borderRadius: "999px",
              ...approvalChipSx,
            }}
          />
          <Box sx={{ flex: 1 }} />
          <Button
            disableRipple
            variant="outlined"
            size="small"
            sx={{
              fontSize: 14,
              fontWeight: 400,
              color: GREEN,
              borderColor: GREEN,
              borderRadius: 1,
              px: 3,
              minWidth: 172,
              ...modalButtonHeightSx,
              ...flatOutlinedSx(GREEN, GREEN),
            }}
          >
            MOVE TO CLEARING
          </Button>
          <Button
            disableRipple
            variant="contained"
            color="primary"
            size="small"
            sx={{
              fontSize: 14,
              fontWeight: 400,
              color: "#fff",
              borderRadius: 1,
              px: 4,
              minWidth: 96,
              ...modalButtonHeightSx,
              ...flatContainedSx,
            }}
          >
            SAVE
          </Button>
        </Box>

        <Box
          sx={{
            display: "flex",
            gap: 3,
            flex: 1,
            minHeight: 0,
            alignItems: "stretch",
            overflow: "hidden",
          }}
        >
          {/* Left: check table */}
          <Box
            sx={{
              flex: "1.05 1 0",
              minWidth: 0,
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Box
              sx={{
                display: "flex",
                alignItems: "flex-end",
                gap: 2,
                mb: 1.5,
                px: 0.5,
              }}
            >
              <Box
                sx={{
                  borderBottom: "2px solid",
                  borderColor: "primary.main",
                  pb: 1,
                  px: 1,
                }}
              >
                <Typography
                  sx={{
                    fontSize: 15,
                    fontWeight: 300,
                    color: "primary.main",
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                  }}
                >
                  CHECK
                </Typography>
              </Box>
              <Box sx={{ flex: 1 }} />
              <Button
                disableRipple
                variant="contained"
                color="primary"
                endIcon={<UploadIcon sx={{ fontSize: 28, color: "#fff" }} />}
                sx={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 1.25,
                  px: 2.25,
                  py: 1,
                  mb: 0.5,
                  minWidth: 152,
                  minHeight: 56,
                  color: "#fff",
                  borderRadius: 1,
                  fontWeight: 300,
                  textTransform: "none",
                  boxShadow: "0 1px 4px rgba(0, 0, 0, 0.18)",
                  "&:hover": {
                    boxShadow: "0 1px 4px rgba(0, 0, 0, 0.18)",
                    bgcolor: "primary.main",
                  },
                  "&:active": {
                    boxShadow: "0 1px 4px rgba(0, 0, 0, 0.18)",
                    bgcolor: "primary.main",
                  },
                  "& .MuiButton-endIcon": {
                    ml: 0.75,
                    mr: 0,
                    alignSelf: "center",
                  },
                }}
              >
                <Box
                  component="span"
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    fontSize: 14,
                    fontWeight: 300,
                    lineHeight: 1.25,
                    letterSpacing: "0.04em",
                    textTransform: "uppercase",
                    whiteSpace: "normal",
                    textAlign: "left",
                  }}
                >
                  <span>UPLOAD</span>
                  <span>BACKUP</span>
                </Box>
              </Button>
            </Box>

            <Box
              sx={{
                bgcolor: "#fff",
                borderRadius: 1,
                border: "1px solid #e0e0e0",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <Box
                sx={{
                  px: 2,
                  py: 1,
                  display: "flex",
                  alignItems: "center",
                  gap: 2.5,
                  flexWrap: "wrap",
                  borderBottom: "1px solid #f0f0f0",
                }}
              >
                {TOOLBAR_ITEMS.map(({ label, icon: Icon }) => (
                  <Button
                    key={label}
                    disableRipple
                    size="small"
                    startIcon={
                      <Icon sx={{ fontSize: 18, color: "primary.main" }} />
                    }
                    sx={{
                      fontSize: 13,
                      fontWeight: 400,
                      color: "primary.main",
                      minWidth: 0,
                      px: 0,
                      ...modalButtonHeightSx,
                      "&:hover": { bgcolor: "transparent" },
                    }}
                  >
                    {label}
                  </Button>
                ))}
              </Box>

              <TableContainer sx={tableScrollSx}>
                <Table size="small" stickyHeader sx={{ minWidth: 960 }}>
                  <TableHead>
                    <TableRow>
                      {[
                        "Invoice Date",
                        "Invoice Nbr",
                        "PO Nbr",
                        "Discount Amt",
                        "Invoice Amt",
                        "Check Amt",
                        "Voucher Nbr",
                      ].map((col) => (
                        <TableCell key={col} sx={tableHeadCellSx}>
                          {col}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {CHECK_ROWS.map((row, i) => (
                      <TableRow
                        key={i}
                        sx={{
                          bgcolor: row.highlight
                            ? alpha("#42a5f5", 0.12)
                            : "transparent",
                        }}
                      >
                        <TableCell sx={tableBodyCellSx}>
                          {row.invoiceDate}
                        </TableCell>
                        <TableCell sx={tableBodyCellSx}>
                          {row.invoiceNbr}
                        </TableCell>
                        <TableCell sx={tableBodyCellSx}>{row.poNbr}</TableCell>
                        <TableCell sx={tableBodyCellSx}>
                          {row.discountAmt}
                        </TableCell>
                        <TableCell sx={tableBodyCellSx}>
                          {row.invoiceAmt}
                        </TableCell>
                        <TableCell sx={tableBodyCellSx}>
                          {row.checkAmt}
                        </TableCell>
                        <TableCell sx={tableBodyCellSx}>
                          {row.voucherNbr}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </Box>

          {/* Right: details + comments */}
          <Box
            sx={{
              flex: "0.95 1 0",
              minWidth: 0,
              maxWidth: "48%",
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
              gap: 2,
              pr: 0.5,
              ...visibleVerticalScrollbarSx,
            }}
          >
            {/* Details */}
            <Box
              sx={{
                bgcolor: "#fff",
                borderRadius: 1,
                border: "1px solid #e8eaed",
                boxShadow: "0 1px 4px rgba(0, 0, 0, 0.06)",
                display: "flex",
                flexDirection: "column",
                flexShrink: 0,
              }}
            >
              <Box
                sx={{
                  px: 2.5,
                  py: 2,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <Typography
                  sx={{ fontWeight: 400, fontSize: 16, color: "#111827" }}
                >
                  Details
                </Typography>
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "stretch",
                    border: "1px solid",
                    borderColor: "primary.main",
                    borderRadius: 1,
                    overflow: "hidden",
                    height: 32,
                  }}
                >
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      px: 3,
                      minWidth: 112,
                      justifyContent: "center",
                    }}
                  >
                    <Typography
                      sx={{
                        fontSize: 12,
                        fontWeight: 400,
                        color: "primary.main",
                        letterSpacing: "0.04em",
                        lineHeight: 1,
                        whiteSpace: "nowrap",
                      }}
                    >
                      IN REVIEW
                    </Typography>
                  </Box>
                  <IconButton
                    size="small"
                    sx={{
                      borderRadius: 0,
                      px: 0.75,
                      borderLeft: "1px solid",
                      borderColor: "primary.main",
                      alignSelf: "stretch",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <ArrowDropDownIcon
                      sx={{ ...buttonChevronSx, color: "primary.main" }}
                    />
                  </IconButton>
                </Box>
              </Box>

              <Box sx={{ p: 2.5, pt: 2 }}>
                {["Deduction Details", "Promotion Details"].map((label) => (
                  <Box key={label} sx={{ ...detailAccordionCardSx, mb: 1.5 }}>
                    <Box sx={detailAccordionHeaderSx}>
                      <KeyboardArrowRightIcon
                        sx={{ fontSize: 20, mr: 1.25, color: LABEL_COLOR }}
                      />
                      <Typography sx={panelTitleSx}>{label}</Typography>
                    </Box>
                  </Box>
                ))}

                {!modalNewDesignEnabled && (
                  <Box sx={{ ...detailAccordionCardSx, mb: 0 }}>
                    <Box
                      onClick={() => setApprovalOpen((o) => !o)}
                      sx={detailAccordionHeaderSx}
                    >
                      {approvalOpen ? (
                        <KeyboardArrowUpIcon
                          sx={{ fontSize: 20, mr: 1.25, color: LABEL_COLOR }}
                        />
                      ) : (
                        <KeyboardArrowRightIcon
                          sx={{ fontSize: 20, mr: 1.25, color: LABEL_COLOR }}
                        />
                      )}
                      <Typography sx={panelTitleSx}>Approval</Typography>
                    </Box>

                    {approvalOpen && (
                      <Box
                        sx={{
                          ...accordionBodyScrollSx,
                          ...approvalPanelScrollSx,
                          pl: `${accordionContentIndentPx}px`,
                          pr: 2,
                          py: 2,
                          pt: 1.5,
                        }}
                      >
                        <Box
                          sx={{
                            display: "grid",
                            gridTemplateColumns: "118px 1fr",
                            columnGap: 3,
                            rowGap: 5,
                            alignItems: "center",
                          }}
                        >
                          <Typography sx={accordionSectionLabelSx}>
                            Assigned to
                          </Typography>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: isReassigning
                                ? "flex-start"
                                : "space-between",
                              gap: 1.5,
                              minWidth: 0,
                              width: "100%",
                            }}
                          >
                            {isReassigning ? (
                              <FormControl
                                size="small"
                                sx={{
                                  flex: 1,
                                  minWidth: 0,
                                  ...reassignSelectFormSx,
                                }}
                              >
                                <Select
                                  fullWidth
                                  displayEmpty
                                  open={reassignMenuOpen}
                                  onOpen={() => setReassignMenuOpen(true)}
                                  onClose={() => setReassignMenuOpen(false)}
                                  value={pendingAssignee}
                                  onChange={(e) =>
                                    setPendingAssignee(e.target.value)
                                  }
                                  IconComponent={ArrowDropDownIcon}
                                  MenuProps={reassignMenuProps}
                                  renderValue={(selected) =>
                                    selected || "\u00a0"
                                  }
                                >
                                  {REASSIGN_ASSIGNEE_OPTIONS.map((option) => (
                                    <MenuItem
                                      key={option}
                                      value={option}
                                      sx={reassignMenuItemSx}
                                    >
                                      {option}
                                    </MenuItem>
                                  ))}
                                </Select>
                              </FormControl>
                            ) : (
                              <Typography
                                sx={{
                                  fontSize: 14,
                                  fontWeight: 400,
                                  color: TABLE_TEXT_COLOR,
                                }}
                              >
                                {assignedToDisplay}
                              </Typography>
                            )}
                            <Button
                              disableRipple
                              variant="outlined"
                              color="primary"
                              size="small"
                              onClick={
                                isReassigning
                                  ? handleAssigneeDone
                                  : handleOpenAssigneeEdit
                              }
                              sx={reassignActionButtonSx}
                            >
                              {isReassigning
                                ? "DONE"
                                : showAssignButton
                                  ? "ASSIGN"
                                  : "REASSIGN"}
                            </Button>
                          </Box>

                          {showAwaitingUserUi ? (
                            <>
                              <Typography
                                sx={{ ...accordionSectionLabelSx, mt: -2.5 }}
                              >
                                Actions
                              </Typography>
                              <Box
                                sx={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: 1,
                                  mt: -2.5,
                                }}
                              >
                                <Button
                                  disableRipple
                                  variant="contained"
                                  size="small"
                                  onClick={onApprove}
                                  sx={modalApproveButtonSx}
                                >
                                  APPROVE
                                </Button>
                                <Button
                                  disableRipple
                                  variant="outlined"
                                  size="small"
                                  onClick={onReject}
                                  sx={modalRejectButtonSx}
                                >
                                  REJECT
                                </Button>
                              </Box>
                            </>
                          ) : null}

                          <Typography sx={accordionSectionLabelSx}>
                            Approval History
                          </Typography>
                          <Box
                            sx={{
                              width: "100%",
                              minWidth: 0,
                              pb: 1.5,
                            }}
                          >
                            <ApprovalHistoryStepper
                              history={approvalHistory}
                              maxWidth="100%"
                            />
                          </Box>
                        </Box>
                      </Box>
                    )}
                  </Box>
                )}
              </Box>
            </Box>

            {modalNewDesignEnabled && (
              <DeductionModalApprovalsAudit
                approvalStatus={approvalStatus}
                approvalHistory={approvalHistory}
                onApprove={onApprove}
                onReject={onReject}
                onReassign={onReassign}
                onUnassign={onUnassign}
              />
            )}

            {!modalNewDesignEnabled && <ModalCommentsSection variant="card" />}
          </Box>
        </Box>

        {/* Footer */}
        <Box
          sx={{
            py: 1,
            mt: 1.5,
            display: "grid",
            gridTemplateColumns: "1fr auto 1fr",
            alignItems: "center",
            bgcolor: "transparent",
            flexShrink: 0,
          }}
        >
          <Button
            disableRipple
            variant="outlined"
            color="primary"
            size="small"
            sx={{
              fontSize: 14,
              fontWeight: 400,
              borderRadius: 1,
              justifySelf: "start",
              ...modalButtonHeightSx,
              ...flatOutlinedSx("primary.main", "primary.main"),
            }}
          >
            PREVIOUS
          </Button>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.5,
              justifySelf: "center",
            }}
          >
            <IosToggle
              checked={modalNewDesignEnabled}
              onChange={setModalNewDesignEnabled}
              aria-label="Toggle modal new design"
            />
            <Button
              disableRipple
              color="primary"
              size="small"
              onClick={handleModalReset}
              sx={modalResetButtonSx}
            >
              RESET
            </Button>
          </Box>
          <Button
            disableRipple
            variant="outlined"
            color="primary"
            size="small"
            sx={{
              fontSize: 14,
              fontWeight: 400,
              borderRadius: 1,
              justifySelf: "end",
              ...modalButtonHeightSx,
              ...flatOutlinedSx("primary.main", "primary.main"),
            }}
          >
            NEXT
          </Button>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
