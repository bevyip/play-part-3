import { useEffect, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import IconButton from "@mui/material/IconButton";
import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";
import { ArrowDropDownIcon } from "../theme/icons";
import ApprovalHistoryStepper from "./ApprovalHistoryStepper";
import { TABLE_TEXT_COLOR } from "../theme/colors";

const PURPLE = "#7b1fa2";
const REJECT_RED = "#d32f2f";
const LABEL_COLOR = "#6b7280";
const REASSIGN_FIELD_FONT_SIZE = 15;
const REASSIGN_LABEL_SHRINK_FONT_SIZE = 13;

const reassignMenuItemSx = {
  fontSize: REASSIGN_FIELD_FONT_SIZE,
};

const REASSIGN_USER_OPTIONS = [
  "Beverly",
  "Kevin",
  "Odette",
  "Matt",
  "Justin Hunter",
  "kevexternal",
];

const containedActionButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  minHeight: 34,
  px: 2.5,
  boxShadow: "none",
  color: "#fff",
  "&:hover": { boxShadow: "none", color: "#fff" },
};

const cancelButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  minHeight: 34,
  px: 2.5,
  boxShadow: "none",
  bgcolor: "#fff",
  color: "primary.main",
  borderColor: "primary.main",
  "&:hover": { bgcolor: "#fff", boxShadow: "none" },
};

const standardSelectSx = {
  color: TABLE_TEXT_COLOR,
  "& .MuiInputBase-input": { fontSize: REASSIGN_FIELD_FONT_SIZE, py: 0.5 },
  "& .MuiInputLabel-root": {
    fontSize: REASSIGN_FIELD_FONT_SIZE,
    color: LABEL_COLOR,
  },
  "& .MuiInputLabel-shrink": {
    fontSize: REASSIGN_LABEL_SHRINK_FONT_SIZE,
  },
  "& .MuiInput-underline:before": { borderBottomColor: "#e0e0e0" },
  "& .MuiInput-underline:hover:not(.Mui-disabled):before": {
    borderBottomColor: "#bdbdbd",
  },
  "& .MuiInput-underline:after": { borderBottomColor: "primary.main" },
  "& .MuiSelect-select": {
    fontSize: REASSIGN_FIELD_FONT_SIZE,
    py: 0.5,
    pr: "24px !important",
  },
  "& .MuiSelect-icon": { fontSize: 20, color: LABEL_COLOR },
};

function getAssigneeName(approvalStatus) {
  if (!approvalStatus?.startsWith("Assigned to ")) return null;
  return approvalStatus.slice("Assigned to ".length);
}

function isAwaitingCurrentUserApproval(approvalStatus, currentUser) {
  return getAssigneeName(approvalStatus) === currentUser;
}

function isUnassignedStatus(approvalStatus) {
  return !approvalStatus || approvalStatus === "Unassigned";
}

function ReassignSelect({
  label,
  value,
  onChange,
  options,
  sx,
  excludeValue,
}) {
  const [open, setOpen] = useState(false);
  const hasValue = Boolean(value);
  const filteredOptions = excludeValue
    ? options.filter((option) => option !== excludeValue)
    : options;

  return (
    <FormControl variant="standard" sx={sx}>
      <InputLabel shrink={hasValue || open}>{label}</InputLabel>
      <Select
        value={value}
        label={label}
        displayEmpty
        open={open}
        onOpen={() => setOpen(true)}
        onClose={() => setOpen(false)}
        onChange={onChange}
        IconComponent={ArrowDropDownIcon}
        renderValue={(selected) => selected || "\u00a0"}
      >
        <MenuItem value="" sx={reassignMenuItemSx}>
          {"\u00a0"}
        </MenuItem>
        {filteredOptions.map((option) => (
          <MenuItem key={option} value={option} sx={reassignMenuItemSx}>
            {option}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}

const reassignConfirmedFieldSx = {
  ...standardSelectSx,
  "& .MuiInputBase-input.Mui-disabled": {
    WebkitTextFillColor: TABLE_TEXT_COLOR,
    color: TABLE_TEXT_COLOR,
  },
  "& .MuiInputLabel-root.Mui-disabled": { color: LABEL_COLOR },
};

function ReassignConfirmedField({ label, value, sx }) {
  return (
    <TextField
      label={label}
      value={value}
      variant="standard"
      disabled
      InputLabelProps={{ shrink: true }}
      sx={{ ...reassignConfirmedFieldSx, ...sx }}
    />
  );
}

const REASSIGN_FROM_FIELD_WIDTH = 240;
const REASSIGN_TO_FIELD_WIDTH = 200;

const reassignFromFieldSx = {
  ...standardSelectSx,
  width: REASSIGN_FROM_FIELD_WIDTH,
  maxWidth: REASSIGN_FROM_FIELD_WIDTH,
};

const reassignToFieldSx = {
  ...standardSelectSx,
  width: REASSIGN_TO_FIELD_WIDTH,
  maxWidth: REASSIGN_TO_FIELD_WIDTH,
  flexShrink: 0,
};

const reassignIconButtonSx = {
  width: 28,
  height: 28,
  mb: 0.5,
  flexShrink: 0,
  "&:hover": { opacity: 0.9 },
};

export default function ViewApprovalModal({
  open,
  onClose,
  approvalStatus,
  canApprove = false,
  approvalHistory,
  onApprove,
  onReject,
  onReassign,
  currentUser = "Beverly",
}) {
  const [comment, setComment] = useState("");
  const [showReassignForm, setShowReassignForm] = useState(false);
  const [reassignFrom, setReassignFrom] = useState("");
  const [reassignToRows, setReassignToRows] = useState([""]);
  const [reassignConfirmed, setReassignConfirmed] = useState(false);

  const showReassignFrom =
    isUnassignedStatus(approvalStatus) ||
    isAwaitingCurrentUserApproval(approvalStatus, currentUser);
  const toFieldLabel = showReassignFrom ? "To" : "Reassign To";
  const toFieldSx = showReassignFrom ? reassignToFieldSx : reassignFromFieldSx;
  const allReassignFieldsFilled = showReassignFrom
    ? Boolean(reassignFrom) && reassignToRows.every(Boolean)
    : reassignToRows.every(Boolean);
  const canConfirmReassign = allReassignFieldsFilled && !reassignConfirmed;

  useEffect(() => {
    if (!open) {
      setComment("");
      setShowReassignForm(false);
      setReassignFrom("");
      setReassignToRows([""]);
      setReassignConfirmed(false);
    }
  }, [open]);

  const resetReassignForm = () => {
    setShowReassignForm(false);
    setReassignFrom("");
    setReassignToRows([""]);
    setReassignConfirmed(false);
  };

  const handleClose = () => {
    resetReassignForm();
    setComment("");
    onClose();
  };

  const handleApprove = () => {
    onApprove?.(comment);
    setComment("");
    resetReassignForm();
  };

  const handleReject = () => {
    onReject?.(comment);
    setComment("");
    resetReassignForm();
  };

  const handleConfirmReassign = () => {
    if (!canConfirmReassign) return;

    const from = showReassignFrom
      ? reassignFrom
      : getAssigneeName(approvalStatus) || "";
    onReassign?.({ from, to: reassignToRows.filter(Boolean) });
    resetReassignForm();
  };

  const updateReassignToRow = (index, value) => {
    setReassignToRows((rows) =>
      rows.map((row, rowIndex) => (rowIndex === index ? value : row)),
    );
  };

  const removeReassignToRow = (index) => {
    if (reassignConfirmed) return;
    setReassignToRows((rows) =>
      rows.length === 1
        ? [""]
        : rows.filter((_, rowIndex) => rowIndex !== index),
    );
  };

  const openReassignForm = () => {
    setReassignFrom("");
    setReassignToRows([""]);
    setReassignConfirmed(false);
    setShowReassignForm(true);
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 1,
          maxWidth: 780,
        },
      }}
    >
      <DialogContent sx={{ px: 5, pt: 5, pb: 5 }}>
        <Typography
          sx={{
            fontSize: 20,
            fontWeight: 400,
            color: "#111827",
            mb: 3,
          }}
        >
          View Approval
        </Typography>

        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            width: "100%",
            py: 3,
          }}
        >
          <ApprovalHistoryStepper
            history={approvalHistory}
            maxWidth={720}
            preventTextWrap
          />
        </Box>

        {canApprove && !showReassignForm ? (
          <TextField
            fullWidth
            multiline
            minRows={1}
            placeholder="Comment (optional)"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            sx={{
              mt: 1,
              mb: 4,
              "& .MuiOutlinedInput-root": {
                fontSize: 14,
                borderRadius: 1,
                alignItems: "flex-start",
                "& fieldset": { borderColor: "#d1d5db" },
                "&:hover fieldset": { borderColor: "#9ca3af" },
                "&.Mui-focused fieldset": { borderColor: "primary.main" },
              },
              "& .MuiOutlinedInput-input": {
                py: 1.25,
              },
              "& .MuiOutlinedInput-input::placeholder": {
                color: "#9ca3af",
                opacity: 1,
              },
            }}
          />
        ) : !showReassignForm ? (
          <Box sx={{ mb: 4 }} />
        ) : null}

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 2,
            flexWrap: "wrap",
            mb: showReassignForm ? 3 : 0,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
            {!showReassignForm ? (
              <>
                <Button
                  disableRipple
                  variant="contained"
                  color="primary"
                  onClick={handleApprove}
                  sx={{
                    ...containedActionButtonSx,
                    "&:hover": {
                      bgcolor: "primary.main",
                      boxShadow: "none",
                      color: "#fff",
                    },
                  }}
                >
                  APPROVE
                </Button>
                <Button
                  disableRipple
                  variant="contained"
                  onClick={handleReject}
                  sx={{
                    ...containedActionButtonSx,
                    bgcolor: REJECT_RED,
                    "&:hover": {
                      bgcolor: REJECT_RED,
                      boxShadow: "none",
                      color: "#fff",
                    },
                  }}
                >
                  REJECT
                </Button>
              </>
            ) : null}
            <Button
              disableRipple
              variant="contained"
              onClick={showReassignForm ? undefined : openReassignForm}
              sx={{
                ...containedActionButtonSx,
                bgcolor: PURPLE,
                cursor: showReassignForm ? "default" : "pointer",
                "&:hover": {
                  bgcolor: PURPLE,
                  boxShadow: "none",
                  color: "#fff",
                  cursor: showReassignForm ? "default" : "pointer",
                },
              }}
            >
              REASSIGN
            </Button>
          </Box>
          <Button
            disableRipple
            variant="outlined"
            color="primary"
            onClick={showReassignForm ? resetReassignForm : handleClose}
            sx={cancelButtonSx}
          >
            CANCEL
          </Button>
        </Box>

        {showReassignForm ? (
          <Box>
            {showReassignFrom ? (
              reassignConfirmed ? (
                <ReassignConfirmedField
                  label="Reassign From"
                  value={reassignFrom}
                  sx={{ ...reassignFromFieldSx, mb: 3 }}
                />
              ) : (
                <ReassignSelect
                  label="Reassign From"
                  value={reassignFrom}
                  onChange={(e) => setReassignFrom(e.target.value)}
                  options={REASSIGN_USER_OPTIONS}
                  sx={{ ...reassignFromFieldSx, mb: 3 }}
                />
              )
            ) : null}

            {reassignToRows.map((rowValue, index) => (
              <Box
                key={`reassign-to-${index}`}
                sx={{
                  display: "flex",
                  alignItems: "flex-end",
                  gap: 1.5,
                  mb: index < reassignToRows.length - 1 ? 2.5 : 0,
                }}
              >
                {reassignConfirmed ? (
                  <ReassignConfirmedField
                    label={toFieldLabel}
                    value={rowValue}
                    sx={toFieldSx}
                  />
                ) : (
                  <ReassignSelect
                    label={toFieldLabel}
                    value={rowValue}
                    onChange={(e) => updateReassignToRow(index, e.target.value)}
                    options={REASSIGN_USER_OPTIONS}
                    excludeValue={
                      showReassignFrom
                        ? reassignFrom
                        : getAssigneeName(approvalStatus)
                    }
                    sx={toFieldSx}
                  />
                )}
                {!reassignConfirmed ? (
                  <>
                    <IconButton
                      disableRipple
                      size="small"
                      onClick={handleConfirmReassign}
                      disabled={!canConfirmReassign}
                      sx={{
                        ...reassignIconButtonSx,
                        bgcolor: canConfirmReassign
                          ? "primary.dark"
                          : "#9ca3af",
                        color: "#fff",
                        cursor: canConfirmReassign ? "pointer" : "default",
                        "&.Mui-disabled": {
                          bgcolor: "#9ca3af",
                          color: "#fff",
                          opacity: 1,
                        },
                      }}
                    >
                      <AddIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                    <IconButton
                      disableRipple
                      size="small"
                      onClick={() => removeReassignToRow(index)}
                      sx={{
                        ...reassignIconButtonSx,
                        bgcolor: REJECT_RED,
                        color: "#fff",
                      }}
                    >
                      <CloseIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </>
                ) : null}
              </Box>
            ))}
          </Box>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
