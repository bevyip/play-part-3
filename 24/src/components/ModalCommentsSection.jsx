import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import IconButton from "@mui/material/IconButton";
import AlternateEmailIcon from "@mui/icons-material/AlternateEmail";
import MoodIcon from "@mui/icons-material/Mood";
import SendIcon from "@mui/icons-material/Send";

const LABEL_COLOR = "#6b7280";

export default function ModalCommentsSection({ variant = "embedded" }) {
  const [commentsTab, setCommentsTab] = useState("open");
  const isCard = variant === "card";

  return (
    <Box
      sx={
        isCard
          ? {
              bgcolor: "#fff",
              borderRadius: 1,
              border: "1px solid #e8eaed",
              boxShadow: "0 1px 4px rgba(0, 0, 0, 0.06)",
              display: "flex",
              flexDirection: "column",
              flexShrink: 0,
            }
          : {
              display: "flex",
              flexDirection: "column",
              border: "1px solid #e8eaed",
              borderRadius: 1,
            }
      }
    >
      {isCard ? (
        <Typography
          sx={{
            px: 2.5,
            pt: 2.5,
            pb: 1.25,
            fontWeight: 400,
            fontSize: 16,
            color: "#111827",
          }}
        >
          Comments
        </Typography>
      ) : null}

      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          gap: 4,
          borderBottom: isCard ? "1px solid #f0f0f0" : "none",
          px: isCard ? 2.5 : 2,
          pt: 2,
          pb: 2,
        }}
      >
        {[
          { id: "open", label: "OPEN" },
          { id: "resolved", label: "RESOLVED" },
        ].map((tab) => (
          <Box
            key={tab.id}
            onClick={() => setCommentsTab(tab.id)}
            sx={{
              borderBottom: "2px solid",
              borderColor:
                commentsTab === tab.id ? "primary.main" : "transparent",
              pb: 1.5,
              cursor: "pointer",
            }}
          >
            <Typography
              sx={{
                fontSize: 14,
                fontWeight: 400,
                color: commentsTab === tab.id ? "primary.main" : LABEL_COLOR,
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
          bgcolor: "#f3f4f6",
          borderRadius: isCard ? 1 : 0,
          py: 3,
          mb: 2,
          mx: isCard ? 2.5 : 0,
          textAlign: "center",
        }}
      >
        <Typography sx={{ fontSize: 14, fontWeight: 400, color: LABEL_COLOR }}>
          No comments yet
        </Typography>
      </Box>

      <Box
        sx={{
          px: isCard ? 2.5 : 2,
          pb: isCard ? 2.5 : 2,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Box
          sx={{
            borderBottom: "1px solid #e0e0e0",
            pb: 1,
            mb: 1,
          }}
        >
          <TextField
            placeholder="Write a comment..."
            variant="standard"
            fullWidth
            multiline
            minRows={2}
            InputProps={{ disableUnderline: true }}
            sx={{
              "& .MuiInputBase-root": { alignItems: "flex-start" },
              "& .MuiInputBase-input": {
                fontSize: 14,
                fontWeight: 400,
                p: 0,
              },
              "& .MuiInputBase-input::placeholder": {
                color: "#9ca3af",
                opacity: 1,
              },
            }}
          />
        </Box>

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box sx={{ display: "flex", gap: 0.25 }}>
            <IconButton size="small" sx={{ p: 0.75 }}>
              <AlternateEmailIcon sx={{ fontSize: 20, color: LABEL_COLOR }} />
            </IconButton>
            <IconButton size="small" sx={{ p: 0.75 }}>
              <MoodIcon sx={{ fontSize: 20, color: LABEL_COLOR }} />
            </IconButton>
          </Box>
          <IconButton size="small" sx={{ p: 0.75 }}>
            <SendIcon sx={{ fontSize: 20, color: "#9ca3af" }} />
          </IconButton>
        </Box>
      </Box>
    </Box>
  );
}
