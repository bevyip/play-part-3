import { useEffect, useId, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Checkbox from "@mui/material/Checkbox";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import Link from "@mui/material/Link";
import { KeyboardArrowDownIcon } from "../theme/icons";
import { TABLE_TEXT_COLOR } from "../theme/colors";
import {
  FALLBACK_USER,
  FALLBACK_USER_EMAIL,
  REASSIGN_ASSIGNEE_OPTIONS,
  UNASSIGN_REASONS,
  REJECT_DEDUCTION_STATUS_OPTIONS,
} from "../utils/approvalHelpers";

const LABEL_COLOR = "#6b7280";
const DESCRIPTION_COLOR = "#616161";
const REQUIRED_ASTERISK_COLOR = "#d32f2f";
const DISABLED_BG = "#e0e0e0";
const MODAL_MAX_WIDTH = 720;
const BODY_FONT_SIZE = 14;
const SECTION_SPACING = 2.5;
const OUTLINED_BORDER_COLOR = "#d1d5db";
const OUTLINED_BORDER_HOVER_COLOR = "#9ca3af";

const outlinedInputRootBorderSx = {
  borderRadius: 1,
  "& fieldset": { borderColor: OUTLINED_BORDER_COLOR },
  "&:hover fieldset": { borderColor: OUTLINED_BORDER_HOVER_COLOR },
  "&.Mui-focused fieldset": { borderColor: "primary.main" },
};

const outlinedSelectBorderSx = {
  borderRadius: 1,
  "& .MuiOutlinedInput-notchedOutline": {
    borderColor: OUTLINED_BORDER_COLOR,
  },
  "&:hover .MuiOutlinedInput-notchedOutline": {
    borderColor: OUTLINED_BORDER_HOVER_COLOR,
  },
  "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
    borderColor: "primary.main",
  },
};

const modalTitleSx = {
  fontSize: 18,
  fontWeight: 700,
  color: "#111827",
  lineHeight: 1.2,
};

const descriptionSx = {
  fontSize: BODY_FONT_SIZE,
  fontWeight: 400,
  color: DESCRIPTION_COLOR,
  lineHeight: 1.55,
  mb: SECTION_SPACING,
};

const requiredAsteriskSx = {
  "& .MuiInputLabel-asterisk": {
    color: REQUIRED_ASTERISK_COLOR,
  },
  "& .MuiFormLabel-asterisk": {
    color: REQUIRED_ASTERISK_COLOR,
  },
};

const commentsLabelSx = {
  display: "block",
  fontSize: BODY_FONT_SIZE,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
  mb: 1,
  ...requiredAsteriskSx,
};

const commentsFieldSx = {
  "& .MuiOutlinedInput-root": {
    fontSize: BODY_FONT_SIZE,
    fontWeight: 400,
    alignItems: "flex-start",
    px: 1.75,
    py: 1.25,
    ...outlinedInputRootBorderSx,
  },
  "& .MuiOutlinedInput-input": {
    py: 0,
    px: 0,
    fontSize: BODY_FONT_SIZE,
    fontWeight: 400,
    lineHeight: 1.55,
  },
  "& .MuiOutlinedInput-input::placeholder": {
    color: "#9ca3af",
    opacity: 1,
    fontSize: BODY_FONT_SIZE,
    fontWeight: 300,
  },
};

const requiredInputLabelSx = {
  fontSize: BODY_FONT_SIZE,
  color: LABEL_COLOR,
  ...requiredAsteriskSx,
};

const selectFormControlSx = {
  mb: SECTION_SPACING,
  ...requiredAsteriskSx,
};

const selectSx = {
  fontSize: BODY_FONT_SIZE,
  color: TABLE_TEXT_COLOR,
  ...outlinedSelectBorderSx,
};

const menuItemSx = {
  fontSize: BODY_FONT_SIZE,
  fontWeight: 400,
  color: TABLE_TEXT_COLOR,
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

const confirmButtonSx = (enabled) => ({
  fontSize: 13,
  fontWeight: 400,
  minHeight: 34,
  px: 2.5,
  boxShadow: "none",
  color: "#fff",
  bgcolor: enabled ? "primary.main" : DISABLED_BG,
  "&:hover": {
    boxShadow: "none",
    bgcolor: enabled ? "primary.dark" : DISABLED_BG,
  },
  "&.Mui-disabled": {
    bgcolor: DISABLED_BG,
    color: "#fff",
  },
});

const MODAL_SEGMENT_BORDER = "#000";
const MODAL_SEGMENT_ACTIVE_BG = "#424242";
const MODAL_ACTION_CONTROL_HEIGHT = 34;

const modalSegmentTextSx = {
  fontSize: 11,
  fontWeight: 400,
  lineHeight: 1.2,
};

const modalSegmentButtonSx = (active) => ({
  ...modalSegmentTextSx,
  border: "none",
  m: 0,
  px: 1,
  height: "100%",
  minWidth: 0,
  whiteSpace: "nowrap",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  fontFamily: "inherit",
  boxSizing: "border-box",
  color: active ? "#fff" : "#111827",
  bgcolor: active ? MODAL_SEGMENT_ACTIVE_BG : "transparent",
  transition: "background-color 0.15s ease, color 0.15s ease",
  "&:focus-visible": {
    outline: "2px solid",
    outlineColor: MODAL_SEGMENT_BORDER,
    outlineOffset: -2,
  },
});

const UNASSIGN_FALLBACK_OPTIONS = [
  { id: "fallback", label: "Fallback User", useFallbackUser: true },
  { id: "no-fallback", label: "No Fallback User", useFallbackUser: false },
];

function UnassignFallbackSegmentedToggle({ useFallbackUser, onChange }) {
  return (
    <Box
      role="group"
      aria-label="Unassign fallback recipient"
      sx={{
        display: "inline-flex",
        alignItems: "stretch",
        height: MODAL_ACTION_CONTROL_HEIGHT,
        minHeight: MODAL_ACTION_CONTROL_HEIGHT,
        border: "1px solid",
        borderColor: MODAL_SEGMENT_BORDER,
        borderRadius: 1,
        overflow: "hidden",
        flexShrink: 0,
        bgcolor: "#fff",
        boxSizing: "border-box",
      }}
    >
      {UNASSIGN_FALLBACK_OPTIONS.map((option, index) => {
        const isActive = useFallbackUser === option.useFallbackUser;

        return (
          <Box
            key={option.id}
            component="button"
            type="button"
            aria-pressed={isActive}
            onClick={() => onChange?.(option.useFallbackUser)}
            sx={{
              ...modalSegmentButtonSx(isActive),
              borderLeft: index > 0 ? "1px solid" : "none",
              borderColor: index > 0 ? MODAL_SEGMENT_BORDER : undefined,
            }}
          >
            {option.label}
          </Box>
        );
      })}
    </Box>
  );
}

const REASSIGN_USER_OPTIONS = REASSIGN_ASSIGNEE_OPTIONS.filter(
  (option) => option !== "Unassign",
);

const ACTION_TITLES = {
  reject: "Reject",
  unassign: "Unassign",
  reassign: "Reassign",
};

function getInitialFormState(actionType) {
  if (actionType === "reject") {
    return {
      deductionStatus: REJECT_DEDUCTION_STATUS_OPTIONS[0],
      reason: "",
      assignee: "",
      updateSalesOwner: false,
      comment: "",
    };
  }
  if (actionType === "unassign") {
    return {
      deductionStatus: "",
      reason: "",
      assignee: "",
      updateSalesOwner: false,
      comment: "",
      useFallbackUser: true,
    };
  }
  return {
    deductionStatus: "",
    reason: "",
    assignee: "",
    updateSalesOwner: false,
    comment: "",
  };
}

function isConfirmEnabled(actionType, form) {
  if (actionType === "reject") {
    return Boolean(form.deductionStatus && form.comment.trim());
  }
  if (actionType === "unassign") {
    return Boolean(form.reason && form.comment.trim());
  }
  if (actionType === "reassign") {
    return Boolean(form.assignee);
  }
  return false;
}

function ActionSelectField({
  label,
  value,
  onChange,
  options,
  required = true,
  sx,
}) {
  const fieldId = useId();
  const labelId = `${fieldId}-label`;

  return (
    <FormControl
      fullWidth
      required={required}
      sx={{ ...selectFormControlSx, ...sx }}
    >
      <InputLabel id={labelId} sx={requiredInputLabelSx}>
        {label}
      </InputLabel>
      <Select
        labelId={labelId}
        id={fieldId}
        value={value}
        label={label}
        onChange={onChange}
        IconComponent={KeyboardArrowDownIcon}
        sx={selectSx}
      >
        {options.map((option) => (
          <MenuItem key={option} value={option} sx={menuItemSx}>
            {option}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}

export default function ApprovalActionModal({
  open,
  actionType,
  onClose,
  onConfirm,
}) {
  const [form, setForm] = useState(() => getInitialFormState(actionType));

  useEffect(() => {
    if (open && actionType) {
      setForm(getInitialFormState(actionType));
    }
  }, [open, actionType]);

  if (!actionType) return null;

  const confirmEnabled = isConfirmEnabled(actionType, form);
  const commentRequired = actionType === "reject" || actionType === "unassign";

  const handleClose = () => {
    onClose?.();
  };

  const handleConfirm = () => {
    if (!confirmEnabled) return;

    if (actionType === "reject") {
      onConfirm?.({
        deductionStatus: form.deductionStatus,
        comment: form.comment.trim(),
      });
    } else if (actionType === "unassign") {
      onConfirm?.({
        reason: form.reason,
        comment: form.comment.trim(),
      });
    } else if (actionType === "reassign") {
      onConfirm?.({
        assignee: form.assignee,
        updateSalesOwner: form.updateSalesOwner,
        comment: form.comment.trim(),
      });
    }

    handleClose();
  };

  const updateForm = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
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
          maxWidth: MODAL_MAX_WIDTH,
        },
      }}
    >
      <DialogContent sx={{ px: 3.5, pt: 3, pb: 3 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mb: SECTION_SPACING,
          }}
        >
          <Typography sx={modalTitleSx}>{ACTION_TITLES[actionType]}</Typography>
          <IconButton
            disableRipple
            onClick={handleClose}
            sx={{ color: LABEL_COLOR, p: 0.5 }}
            aria-label="Close"
          >
            <CloseIcon sx={{ fontSize: 22 }} />
          </IconButton>
        </Box>

        {actionType === "reject" ? (
          <>
            <Typography sx={descriptionSx}>
              To reject, update the Deduction Status and leave a comment:
            </Typography>
            <ActionSelectField
              label="Deduction Status"
              value={form.deductionStatus}
              onChange={(e) => updateForm("deductionStatus", e.target.value)}
              options={REJECT_DEDUCTION_STATUS_OPTIONS}
            />
          </>
        ) : null}

        {actionType === "unassign" ? (
          <>
            {form.useFallbackUser ? (
              <Typography sx={descriptionSx}>
                This deduction will be unassigned from you and reassigned to{" "}
                <Box
                  component="span"
                  sx={{ fontWeight: 700, color: DESCRIPTION_COLOR }}
                >
                  {FALLBACK_USER} ({FALLBACK_USER_EMAIL})
                </Box>{" "}
                based on Account Settings. Please select a reason and leave a
                comment to explain why it&apos;s being unassigned.
              </Typography>
            ) : (
              <Typography sx={descriptionSx}>
                This deduction will be unassigned from you.{" "}
                <Box
                  component="span"
                  sx={{ fontWeight: 700, color: DESCRIPTION_COLOR }}
                >
                  No fallback recipient has been configured, so the assignment
                  will remain empty.
                </Box>{" "}
                Please select a reason and leave a comment to explain why
                it&apos;s being unassigned.
              </Typography>
            )}
            <ActionSelectField
              label="Reason"
              value={form.reason}
              onChange={(e) => updateForm("reason", e.target.value)}
              options={UNASSIGN_REASONS}
            />
          </>
        ) : null}

        {actionType === "reassign" ? (
          <>
            <Typography sx={descriptionSx}>Reassign to:</Typography>
            <ActionSelectField
              label="User"
              value={form.assignee}
              onChange={(e) => updateForm("assignee", e.target.value)}
              options={REASSIGN_USER_OPTIONS}
            />
            <Box
              sx={{
                display: "flex",
                alignItems: "flex-start",
                gap: 1,
                mb: SECTION_SPACING,
              }}
            >
              <Checkbox
                size="small"
                checked={form.updateSalesOwner}
                onChange={(e) =>
                  updateForm("updateSalesOwner", e.target.checked)
                }
                sx={{ p: 0, mt: 0.125, flexShrink: 0 }}
              />
              <Typography
                sx={{
                  fontSize: BODY_FONT_SIZE,
                  color: TABLE_TEXT_COLOR,
                  lineHeight: 1.55,
                  pt: 0.125,
                }}
              >
                Update the sales owner for Planning Group to this user.
              </Typography>
            </Box>
          </>
        ) : null}

        <Box sx={{ mb: actionType === "unassign" ? 1.5 : SECTION_SPACING }}>
          <FormControl
            fullWidth
            required={commentRequired}
            sx={requiredAsteriskSx}
          >
            <FormLabel required={commentRequired} sx={commentsLabelSx}>
              Comments:
            </FormLabel>
            <TextField
              fullWidth
              multiline
              minRows={4}
              variant="outlined"
              placeholder="Enter your comments here..."
              value={form.comment}
              onChange={(e) => updateForm("comment", e.target.value)}
              sx={commentsFieldSx}
            />
          </FormControl>
        </Box>

        {actionType === "unassign" ? (
          <Typography sx={{ fontSize: 13, color: LABEL_COLOR, mb: 3 }}>
            To update the recipient, go to{" "}
            <Link
              component="span"
              underline="hover"
              sx={{ fontSize: 13, color: "primary.main", cursor: "default" }}
            >
              Account Settings
            </Link>
            .
          </Typography>
        ) : null}

        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 1.5,
          }}
        >
          {actionType === "unassign" ? (
            <UnassignFallbackSegmentedToggle
              useFallbackUser={form.useFallbackUser}
              onChange={(value) => updateForm("useFallbackUser", value)}
            />
          ) : (
            <Box />
          )}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.5,
              ml: "auto",
            }}
          >
            <Button
              disableRipple
              variant="outlined"
              onClick={handleClose}
              sx={cancelButtonSx}
            >
              CANCEL
            </Button>
            <Button
              disableRipple
              variant="contained"
              onClick={handleConfirm}
              disabled={!confirmEnabled}
              sx={confirmButtonSx(confirmEnabled)}
            >
              CONFIRM
            </Button>
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
