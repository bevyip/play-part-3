import {
  CURRENT_USER,
  FALLBACK_USER,
} from "./approvalHelpers";
import {
  getAssigneeChangeHistoryEntry,
  buildUnassignWithFallbackEntries,
  prependApprovalHistoryEntry,
  prependMultipleApprovalHistoryEntries,
} from "../data/approvalHistory";

export function applyRejectManageAction(row, { comment, deductionStatus }) {
  const nextRow = { ...row };

  if (deductionStatus !== undefined) {
    nextRow.deductionStatus = deductionStatus || null;
  }

  return {
    ...nextRow,
    approvalStatus: "Rejected",
    approvedBy: null,
    approvalHistory: prependApprovalHistoryEntry(row.approvalHistory, {
      action: "Rejected at",
      person: CURRENT_USER,
      actor: CURRENT_USER,
      comment,
    }),
  };
}

export function applyReassignManageAction(
  row,
  { assignee, comment },
  actor = CURRENT_USER,
) {
  const historyEntry = getAssigneeChangeHistoryEntry(
    assignee,
    row.approvalStatus,
    actor,
    { comment },
  );

  return {
    ...row,
    approvalStatus: `Assigned to ${assignee}`,
    approvalHistory: prependApprovalHistoryEntry(
      row.approvalHistory,
      historyEntry,
    ),
  };
}

export function applyUnassignManageAction(
  row,
  { reason, comment },
  actor = CURRENT_USER,
  fallbackUser = FALLBACK_USER,
) {
  const historyEntries = buildUnassignWithFallbackEntries({
    reason,
    comment,
    actor,
    fallbackUser,
  });

  return {
    ...row,
    approvalStatus: `Assigned to ${fallbackUser}`,
    approvalHistory: prependMultipleApprovalHistoryEntries(
      row.approvalHistory,
      historyEntries,
    ),
  };
}
