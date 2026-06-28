import { useState } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import Box from "@mui/material/Box";
import Sidebar from "./components/Sidebar";
import DeductionsPage from "./pages/DeductionsPage";
import CashAppPage from "./pages/CashAppPage";

const theme = createTheme({
  palette: {
    primary: { main: "#42a5f5", light: "#64b5f6", dark: "#1e88e5" },
    background: { default: "#f5f5f5" },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", sans-serif',
    fontSize: 13,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: { textTransform: "none", fontWeight: 600, borderRadius: 4 },
      },
    },
  },
});

export default function App() {
  const [activePage, setActivePage] = useState("deductions");
  const [newDesignEnabled, setNewDesignEnabled] = useState(false);
  const [resetKey, setResetKey] = useState(0);

  const handleResetAll = () => {
    setResetKey((key) => key + 1);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: "flex", height: "100vh", overflow: "hidden" }}>
        <Sidebar
          activePage={activePage}
          onNavigate={setActivePage}
          newDesignEnabled={newDesignEnabled}
          onNewDesignChange={setNewDesignEnabled}
          onReset={handleResetAll}
        />
        <Box sx={{ flex: 1, overflow: "auto", bgcolor: "#f5f5f5" }}>
          <Box sx={{ display: activePage === "deductions" ? "block" : "none" }}>
            <DeductionsPage
              newDesignEnabled={newDesignEnabled}
              resetKey={resetKey}
            />
          </Box>
          <Box sx={{ display: activePage === "cashapp" ? "block" : "none" }}>
            <CashAppPage
              newDesignEnabled={newDesignEnabled}
              resetKey={resetKey}
            />
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}
