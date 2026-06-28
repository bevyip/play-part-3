import Box from "@mui/material/Box";

const TRACK_WIDTH = 44;
const TRACK_HEIGHT = 22;
const THUMB_SIZE = 18;
const THUMB_OFFSET = 2;
const THUMB_TRAVEL = TRACK_WIDTH - THUMB_SIZE - THUMB_OFFSET * 2;

export default function IosToggle({ checked = false, onChange, disabled = false, "aria-label": ariaLabel }) {
  return (
    <Box
      component="button"
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      disabled={disabled}
      onClick={(event) => {
        event.stopPropagation();
        if (!disabled) {
          onChange?.(!checked);
        }
      }}
      sx={{
        width: TRACK_WIDTH,
        height: TRACK_HEIGHT,
        borderRadius: `${TRACK_HEIGHT / 2}px`,
        border: "none",
        p: 0,
        flexShrink: 0,
        cursor: disabled ? "default" : "pointer",
        bgcolor: checked ? "#34C759" : "rgba(120, 120, 128, 0.32)",
        opacity: disabled ? 0.5 : 1,
        position: "relative",
        transition: "background-color 0.25s ease-in-out",
        "&:hover": { bgcolor: disabled ? undefined : checked ? "#30B350" : "rgba(120, 120, 128, 0.4)" },
        "&:active": { bgcolor: disabled ? undefined : checked ? "#2DA44E" : "rgba(120, 120, 128, 0.48)" },
        "&:focus-visible": {
          outline: "2px solid rgba(52, 199, 89, 0.6)",
          outlineOffset: 2,
        },
      }}
    >
      <Box
        sx={{
          width: THUMB_SIZE,
          height: THUMB_SIZE,
          borderRadius: "50%",
          bgcolor: "#fff",
          boxShadow: "0 2px 6px rgba(0, 0, 0, 0.15), 0 1px 1px rgba(0, 0, 0, 0.06)",
          position: "absolute",
          top: THUMB_OFFSET,
          left: checked ? THUMB_OFFSET + THUMB_TRAVEL : THUMB_OFFSET,
          transition: "left 0.25s ease-in-out",
        }}
      />
    </Box>
  );
}
