/** Simple rectangular tooltip — no arrow, extra vertical padding. */
export const simpleTooltipProps = {
  placement: "bottom",
  componentsProps: {
    tooltip: {
      sx: {
        py: 1.25,
        px: 1.5,
        fontSize: 11,
        fontWeight: 400,
        lineHeight: 1.35,
        borderRadius: 1,
        bgcolor: "rgba(97, 97, 97, 0.92)",
      },
    },
  },
};
