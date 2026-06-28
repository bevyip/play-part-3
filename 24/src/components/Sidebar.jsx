import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Collapse from "@mui/material/Collapse";
import { ArrowDropDownIcon, ArrowDropUpIcon, buttonChevronSx } from "../theme/icons";
import { useState } from "react";
import confidoLogo from "../img/confido-logo.png";
import IosToggle from "./IosToggle";

const MAIN_NAV = [
  { label: "Home", id: "home", inert: true },
  {
    label: "Accounting",
    id: "accounting",
    children: [
      { label: "Cash Application", id: "cashapp" },
      { label: "Deductions", id: "deductions" },
      { label: "Deductions Reports", id: "reports", inert: true },
      { label: "Clearing", id: "clearing", inert: true },
      { label: "Disputes", id: "disputes", inert: true },
      { label: "Audit Logs", id: "audit", inert: true },
    ],
  },
  { label: "Trade", id: "trade", hasChildren: true, inert: true },
  { label: "Sales", id: "sales", hasChildren: true, inert: true },
  {
    label: "Automation Tools",
    id: "automation",
    hasChildren: true,
    inert: true,
  },
];

const BOTTOM_NAV = [
  { label: "Settings", id: "settings", inert: true },
  { label: "Integrations", id: "integrations", inert: true },
  { label: "Products", id: "products", inert: true },
  { label: "Team", id: "team", inert: true },
];

const ACTIVE_BG = "rgba(255,255,255,0.08)";

const mainTitleProps = { fontSize: 15, fontWeight: 600, color: "#fff" };

const subLinkProps = {
  fontSize: 14.5,
  fontWeight: 400,
  color: "rgba(255,255,255,0.65)",
};

const chevronSx = { ...buttonChevronSx, color: "rgba(255,255,255,0.7)" };

const sidebarResetButtonSx = {
  fontSize: 13,
  fontWeight: 400,
  color: "#42a5f5",
  p: 0,
  minWidth: 0,
  minHeight: 0,
  lineHeight: 1,
  "&:hover": { bgcolor: "transparent", color: "#64b5f6" },
};

const navButtonSx = {
  px: 2.5,
  py: 0.75,
  "&:hover": { bgcolor: "transparent" },
  "&:active": { bgcolor: "transparent" },
  "&.Mui-focusVisible": { bgcolor: "transparent", outline: "none" },
};

export default function Sidebar({
  activePage,
  onNavigate,
  newDesignEnabled = false,
  onNewDesignChange,
  onReset,
}) {
  const [openSections, setOpenSections] = useState({
    accounting: true,
  });

  const toggleSection = (sectionId) => {
    setOpenSections((prev) => ({ ...prev, [sectionId]: !prev[sectionId] }));
  };

  const renderNavItem = (item) => {
    if (item.children) {
      const isOpen = openSections[item.id];

      return (
        <Box key={item.id}>
          <ListItemButton
            disableRipple
            onClick={() => toggleSection(item.id)}
            sx={{ ...navButtonSx }}
          >
            <ListItemText
              primary={item.label}
              primaryTypographyProps={mainTitleProps}
            />
            {isOpen ? (
              <ArrowDropUpIcon sx={chevronSx} />
            ) : (
              <ArrowDropDownIcon sx={chevronSx} />
            )}
          </ListItemButton>
          <Collapse in={isOpen}>
            {item.children.map((child) => {
              const isActive = activePage === child.id;
              return (
                <ListItemButton
                  key={child.id}
                  disableRipple
                  onClick={child.inert ? undefined : () => onNavigate(child.id)}
                  sx={{
                    ...navButtonSx,
                    pl: 3.5,
                    py: 0.6,
                    cursor: "pointer",
                    bgcolor: isActive ? ACTIVE_BG : "transparent",
                    transition: "background-color 0.25s ease-in-out",
                    ...(isActive && {
                      "&:hover": { bgcolor: ACTIVE_BG },
                      "&:active": { bgcolor: ACTIVE_BG },
                      "&.Mui-focusVisible": {
                        bgcolor: ACTIVE_BG,
                        outline: "none",
                      },
                    }),
                  }}
                >
                  <ListItemText
                    primary={child.label}
                    primaryTypographyProps={subLinkProps}
                  />
                </ListItemButton>
              );
            })}
          </Collapse>
        </Box>
      );
    }

    return (
      <ListItemButton
        key={item.id}
        disableRipple
        onClick={item.inert ? undefined : () => onNavigate(item.id)}
        sx={{
          ...navButtonSx,
          cursor: "pointer",
        }}
      >
        <ListItemText
          primary={item.label}
          primaryTypographyProps={mainTitleProps}
        />
        {item.hasChildren && <ArrowDropDownIcon sx={chevronSx} />}
      </ListItemButton>
    );
  };

  return (
    <Box
      sx={{
        width: 220,
        minWidth: 220,
        bgcolor: "#2d2d2d",
        color: "#fff",
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      <Box sx={{ px: 2, py: 2, display: "flex", justifyContent: "center" }}>
        <Box
          component="img"
          src={confidoLogo}
          alt="Confido"
          sx={{ width: "88%", maxWidth: 170, height: "auto", display: "block" }}
        />
      </Box>

      <List dense disablePadding sx={{ flex: 1, py: 1, overflow: "auto" }}>
        {MAIN_NAV.map(renderNavItem)}
      </List>

      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          px: 2,
          pb: 1.25,
          pt: 0.5,
        }}
      >
        <IosToggle
          checked={newDesignEnabled}
          onChange={onNewDesignChange}
          aria-label="Toggle new design"
        />
        <Button
          disableRipple
          onClick={onReset}
          sx={sidebarResetButtonSx}
        >
          RESET
        </Button>
      </Box>

      <List
        dense
        disablePadding
        sx={{
          py: 1,
          borderTop: "0.5px solid rgba(255,255,255,0.5)",
        }}
      >
        {BOTTOM_NAV.map(renderNavItem)}
      </List>
    </Box>
  );
}
