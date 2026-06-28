import {
  TABLE_TEXT_COLOR,
  AWAITING_APPROVAL_ORANGE,
  PENDING_APPROVAL_BLUE,
} from "../theme/colors";
import {
  NEW_DESIGN_APPROVE_GREEN,
  NEW_DESIGN_REJECT_RED,
} from "../theme/newDesignActionButtons";

export const CURRENT_USER = "Beverly";
export const FALLBACK_USER = "Beverly";
export const FALLBACK_USER_EMAIL = "beverly@confidotech.com";
export const UNASSIGN_REASONS = ["Misassigned", "Other"];
export const REJECT_DEDUCTION_STATUS_OPTIONS = ["In Review"];

export function getAssigneeName(approvalStatus) {
  if (!approvalStatus?.startsWith("Assigned to ")) return null;
  return approvalStatus.slice("Assigned to ".length);
}

export function isAwaitingCurrentUserApproval(approvalStatus) {
  return getAssigneeName(approvalStatus) === CURRENT_USER;
}

export function isAssignedToOtherUser(approvalStatus) {
  const assignee = getAssigneeName(approvalStatus);
  return Boolean(assignee && assignee !== CURRENT_USER);
}

export function isUnassignedStatus(approvalStatus) {
  if (approvalStatus == null || approvalStatus === "") return true;
  return String(approvalStatus).trim() === "Unassigned";
}

export function getAssignmentDisplay(approvalStatus) {
  if (isUnassignedStatus(approvalStatus)) return "Unassigned";
  return getAssigneeName(approvalStatus) || "Unassigned";
}

export function getApprovalStatusDisplayWithColor(approvalStatus) {
  if (isUnassignedStatus(approvalStatus)) {
    return { text: "Unassigned", color: TABLE_TEXT_COLOR };
  }
  if (isAwaitingCurrentUserApproval(approvalStatus)) {
    return {
      text: "Awaiting your Approval",
      color: AWAITING_APPROVAL_ORANGE,
    };
  }
  if (isAssignedToOtherUser(approvalStatus)) {
    return { text: "Pending Approval", color: PENDING_APPROVAL_BLUE };
  }
  if (approvalStatus === "Approved") {
    return { text: "Approved", color: NEW_DESIGN_APPROVE_GREEN };
  }
  if (approvalStatus === "Rejected") {
    return { text: "Rejected", color: NEW_DESIGN_REJECT_RED };
  }
  return {
    text: approvalStatus || "",
    color: TABLE_TEXT_COLOR,
  };
}

export function getApprovalStatusDisplay(approvalStatus) {
  return getApprovalStatusDisplayWithColor(approvalStatus).text;
}

export function canReassignFromManageMenu(approvalStatus) {
  if (isAwaitingCurrentUserApproval(approvalStatus)) return true;
  if (isAssignedToOtherUser(approvalStatus)) return true;
  if (isUnassignedStatus(approvalStatus)) return true;
  return getAssignmentDisplay(approvalStatus) === "Unassigned";
}

export function getManageMenuOptions(approvalStatus) {
  const assignedToMe = isAwaitingCurrentUserApproval(approvalStatus);
  const assignedToOther = isAssignedToOtherUser(approvalStatus);

  return {
    approve: assignedToMe,
    reject: assignedToMe,
    reassign: canReassignFromManageMenu(approvalStatus),
    unassign: assignedToMe || assignedToOther,
  };
}

export const REASSIGN_ASSIGNEE_OPTIONS = [
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
