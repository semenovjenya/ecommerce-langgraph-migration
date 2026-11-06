from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from commerce_agent.conversation_context import ConversationSummary


class CommerceRuntimeContext(BaseModel):
    """Verified, non-durable identity and policy context supplied by the BFF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: StrictStr = Field(min_length=1)
    user_id: StrictStr = Field(min_length=1)
    roles: tuple[StrictStr, ...] = Field(min_length=1)
    request_id: StrictStr = Field(min_length=1)
    locale: StrictStr = "ru-RU"
    model_profile: StrictStr = "default"
    feature_policy_version: StrictStr = "m1"
    conversation_summary: ConversationSummary | None = Field(default=None,exclude=True,repr=False)

    @field_validator("conversation_summary",mode="before")
    @classmethod
    def trusted_summary_resource(cls,value):
        if value is not None and not isinstance(value,ConversationSummary):
            raise ValueError("conversation summary is a trusted Python resource, not a payload field")
        return value

    @field_validator(
        "tenant_id",
        "user_id",
        "request_id",
        "locale",
        "model_profile",
        "feature_policy_version",
    )
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("roles")
    @classmethod
    def reject_blank_roles(cls, roles: tuple[str, ...]) -> tuple[str, ...]:
        if any(not role.strip() for role in roles):
            raise ValueError("roles must not contain blank values")
        return roles
