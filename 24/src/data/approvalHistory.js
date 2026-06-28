import {
  AWAITING_APPROVAL_ORANGE,
  PENDING_APPROVAL_BLUE,
} from "../theme/colors";

/** MUI success.main (accept green) and orange[600] */
export const APPROVAL_HISTORY = [
  {
    action: "Reassigned at",
    date: "8/8/2025, 10:56:00 AM",
    person: "Beverly",
    actor: "Beverly",
    comment: "Reassign for demo",
    color: PENDING_APPROVAL_BLUE,
  },
  {
    action: "Approved at",
    date: "6/25/2025, 10:06:57 AM",
    person: "Beverly",
    actor: "Beverly",
    color: "#2e7d32",
  },
  {
    action: "Assigned at",
    date: "6/25/2025, 11:10:36 AM",
    person: "Beverly",
    actor: "Beverly",
    color: AWAITING_APPROVAL_ORANGE,
  },
  {
    action: "Submitted at",
    date: "6/24/2025, 4:22:08 PM",
    person: "Maria",
    actor: "Maria",
    color: PENDING_APPROVAL_BLUE,
  },
];

const HISTORY_ACTION_COLORS = {
  submitted: PENDING_APPROVAL_BLUE,
  reassigned: PENDING_APPROVAL_BLUE,
  approved: "#2e7d32",
  rejected: "#d32f2f",
  assigned: AWAITING_APPROVAL_ORANGE,
  unassigned: "#616161",
};

export function formatApprovalHistoryTimestamp(date = new Date()) {
  return date.toLocaleString("en-US", {
    month: "numeric",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

export function getHistoryColorForAction(action) {
  if (action.startsWith("Approved")) return HISTORY_ACTION_COLORS.approved;
  if (action.startsWith("Rejected")) return HISTORY_ACTION_COLORS.rejected;
  if (action.startsWith("Assigned")) return HISTORY_ACTION_COLORS.assigned;
  if (action.startsWith("Unassigned")) return HISTORY_ACTION_COLORS.unassigned;
  if (action.startsWith("Reassigned")) return HISTORY_ACTION_COLORS.reassigned;
  return HISTORY_ACTION_COLORS.submitted;
}

export function createApprovalHistoryEntry({
  action,
  person,
  actor,
  comment,
  reason,
}) {
  return {
    action,
    date: formatApprovalHistoryTimestamp(),
    person,
    ...(actor ? { actor } : {}),
    ...(comment ? { comment } : {}),
    ...(reason ? { reason } : {}),
    color: getHistoryColorForAction(action),
  };
}

export function cloneDefaultApprovalHistory() {
  return APPROVAL_HISTORY.map((entry) => ({ ...entry }));
}

export function prependApprovalHistoryEntry(history, entryInput) {
  const entry =
    entryInput.date !== undefined
      ? {
          ...createApprovalHistoryEntry(entryInput),
          date: entryInput.date,
        }
      : createApprovalHistoryEntry(entryInput);
  const base = history?.length ? history : cloneDefaultApprovalHistory();
  return [entry, ...base];
}

export function getAssigneeChangeHistoryEntry(
  assignee,
  previousApprovalStatus,
  actor,
  { comment } = {},
) {
  if (assignee === "Unassign") {
    return {
      action: "Unassigned at",
      person: actor,
      actor,
      ...(comment ? { comment } : {}),
    };
  }

  const wasUnassigned =
    !previousApprovalStatus || previousApprovalStatus === "Unassigned";

  return {
    action: wasUnassigned ? "Assigned at" : "Reassigned at",
    person: assignee,
    actor,
    ...(comment ? { comment } : {}),
  };
}

export function buildUnassignWithFallbackEntries({
  reason,
  comment,
  actor,
  fallbackUser,
}) {
  const timestamp = formatApprovalHistoryTimestamp();

  return [
    {
      action: "Reassigned at",
      person: fallbackUser,
      actor,
      comment: "Automatically assigned to fallback user",
      date: timestamp,
    },
    {
      action: "Unassigned at",
      person: actor,
      actor,
      reason,
      ...(comment ? { comment } : {}),
      date: timestamp,
    },
  ];
}

export function prependMultipleApprovalHistoryEntries(history, entryInputs) {
  const base = history?.length ? history : cloneDefaultApprovalHistory();
  const entries = entryInputs.map((entryInput) => {
    const entry =
      entryInput.date !== undefined
        ? {
            ...createApprovalHistoryEntry(entryInput),
            date: entryInput.date,
          }
        : createApprovalHistoryEntry(entryInput);
    return entry;
  });
  return [...entries, ...base];
}

export function formatAuditLogTimestamp(dateString) {
  if (!dateString) return "";
  const parsed = new Date(dateString);
  if (Number.isNaN(parsed.getTime())) return dateString.toUpperCase();

  return parsed
    .toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    })
    .toUpperCase();
}

export function formatAuditLogPrimaryText(entry) {
  const actor = entry.actor || entry.person || "";
  const action = entry.action?.replace(/ at$/i, "") || "";

  if (action === "Reassigned") {
    return `${actor} · Reassigned to ${entry.person}`;
  }
  if (action === "Assigned") {
    return `${actor} · Assigned to ${entry.person}`;
  }
  if (action === "Unassigned") {
    return `${actor} · Unassigned`;
  }
  if (action === "Approved") {
    return `${actor} · Approved`;
  }
  if (action === "Rejected") {
    return `${actor} · Rejected`;
  }
  if (action === "Submitted") {
    return `${actor} · Submitted`;
  }

  return `${actor} · ${action}`;
}
