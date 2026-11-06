"""Native graph capability: memory checks precede tracing callbacks.

Single-inheritance subclasses preserve the pinned compiler's object layout;
shared implementations explicitly delegate to the existing capability class.
"""
from langgraph.graph.state import CompiledStateGraph
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER,CONFIG_KEY_RUNTIME
from langgraph._internal._config import ensure_config

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.conversation_context import ConversationSummary,capture_conversation,validate_conversation
from commerce_agent.debug_privacy import development_debug_enabled
from commerce_agent.sales.capability_guard import SalesCapabilityGraph
from commerce_agent.marketing.capability_guard import MarketingCapabilityGraph
from commerce_agent.supply.capability_guard import SupplyCapabilityGraph


class ConversationGuard:
    __slots__=()
    def __init__(self,*args,_conversation_enabled=False,_conversation_domain="root",_conversation_workflow="v1",_conversation_workflows=(),**kwargs):
        self._conversation_base.__init__(self,*args,**kwargs)
        self._conversation_enabled=_conversation_enabled
        self._conversation_domain=_conversation_domain
        self._conversation_workflow=_conversation_workflow
        self._conversation_workflows=_conversation_workflows

    def _conversation_check(self,input,effective,kwargs,checkpoint=None):
        enabled=self._conversation_enabled;domain=self._conversation_domain
        if enabled and development_debug_enabled():raise ValueError("conversation source debug is not accepted")
        values=checkpoint.values if checkpoint else {}
        keys=(f"{domain}_conversation",f"{domain}_conversation_version")
        if isinstance(input,dict) and any(key in input for key in keys):
            raise ValueError("conversation snapshot is not invocation input")
        present=any(values.get(key)is not None for key in keys) if domain!="root" else False
        if present and not enabled:raise ValueError("conversation checkpoint capability disabled")
        if not enabled:return
        cfg=effective.get("configurable",{});thread_id=cfg.get("thread_id")
        inherited=cfg.get(CONFIG_KEY_RUNTIME)
        context=kwargs.get("context") or getattr(inherited,"context",None)
        if not isinstance(context,CommerceRuntimeContext):raise ValueError("verified conversation context required")
        if not isinstance(thread_id,str) or not thread_id:raise ValueError("conversation requires logical thread ID")
        resource=context.conversation_summary
        if resource is not None:
            if not isinstance(resource,ConversationSummary):raise ValueError("trusted conversation resource required")
            resource=ConversationSummary.model_validate(resource.model_dump(),strict=True)
            if (resource.thread_id!=thread_id or resource.tenant_id!=context.tenant_id or resource.user_id!=context.user_id):
                raise ValueError("conversation owner/thread mismatch")
        if present:
            if type(values.get(keys[1]))is not int or values[keys[1]]!=1 or not isinstance(values.get(keys[0]),dict):
                raise ValueError("conversation checkpoint markers incomplete")
            validate_conversation(values[keys[0]],values.get("messages",[]),context,thread_id,domain,self._conversation_workflow)
        elif input is None and domain!="root" and (values.get("prepared_"+domain) or values.get(domain+"_candidate")):
            # Keep old threads on their default/disabled reader. An opt-in
            # consumer must not silently enrol an unmarked paused checkpoint.
            raise ValueError("conversation checkpoint source unknown; use legacy reader")
        if input is not None:
            messages=input.get("messages",[]) if isinstance(input,dict) else []
            if domain=="root" and values:
                from commerce_agent.reducers import safe_add_messages
                messages=safe_add_messages(values.get("messages",[]),messages)
            for workflow in self._conversation_workflows or (self._conversation_workflow,):
                capture_conversation(messages,context,thread_id,domain,workflow)

    def _conversation_has_saver(self,effective):
        cfg=effective.get("configurable",{})
        return "thread_id" in cfg and cfg.get(CONFIG_KEY_CHECKPOINTER,self.checkpointer)not in (None,False)

    async def astream(self,input,config=None,**kwargs):
        effective=ensure_config(self.config,config)
        # Debug/invalid config are rejected even before loading checkpoints.
        if self._conversation_enabled and development_debug_enabled():raise ValueError("conversation source debug is not accepted")
        checkpoint=await self._conversation_base.aget_state(self,effective) if self._conversation_has_saver(effective) else None
        self._conversation_check(input,effective,kwargs,checkpoint)
        async for item in self._conversation_base.astream(self,input,config=config,**kwargs):yield item

    def stream(self,input,config=None,**kwargs):
        effective=ensure_config(self.config,config)
        if self._conversation_enabled and development_debug_enabled():raise ValueError("conversation source debug is not accepted")
        checkpoint=self._conversation_base.get_state(self,effective) if self._conversation_has_saver(effective) else None
        self._conversation_check(input,effective,kwargs,checkpoint)
        yield from self._conversation_base.stream(self,input,config=config,**kwargs)


class ConversationGraph(CompiledStateGraph):
    _conversation_base=CompiledStateGraph
    __init__=ConversationGuard.__init__
    _conversation_check=ConversationGuard._conversation_check
    _conversation_has_saver=ConversationGuard._conversation_has_saver
    astream=ConversationGuard.astream
    stream=ConversationGuard.stream


class ConversationSalesGraph(SalesCapabilityGraph):
    _conversation_base=SalesCapabilityGraph
    __init__=ConversationGuard.__init__
    _conversation_check=ConversationGuard._conversation_check
    _conversation_has_saver=ConversationGuard._conversation_has_saver
    astream=ConversationGuard.astream
    stream=ConversationGuard.stream


class ConversationMarketingGraph(MarketingCapabilityGraph):
    _conversation_base=MarketingCapabilityGraph
    __init__=ConversationGuard.__init__
    _conversation_check=ConversationGuard._conversation_check
    _conversation_has_saver=ConversationGuard._conversation_has_saver
    astream=ConversationGuard.astream
    stream=ConversationGuard.stream


class ConversationSupplyGraph(SupplyCapabilityGraph):
    _conversation_base=SupplyCapabilityGraph
    __init__=ConversationGuard.__init__
    _conversation_check=ConversationGuard._conversation_check
    _conversation_has_saver=ConversationGuard._conversation_has_saver
    astream=ConversationGuard.astream
    stream=ConversationGuard.stream


def guard_conversation(compiled,*,enabled,domain,workflow="v1",workflows=()):
    if type(enabled)is not bool:raise ValueError("conversation capability must be bool")
    if enabled and development_debug_enabled():raise ValueError("conversation source debug is not accepted")
    if (not isinstance(workflows,tuple) or any(value not in {"v1","v2_prompt","v2_optimized"}for value in workflows)
        or (workflows and domain!="root")):
        raise ValueError("invalid conversation consumer workflows")
    classes={CompiledStateGraph:ConversationGraph,SalesCapabilityGraph:ConversationSalesGraph,
        MarketingCapabilityGraph:ConversationMarketingGraph,SupplyCapabilityGraph:ConversationSupplyGraph}
    if type(compiled)not in classes:raise TypeError("unexpected conversation graph compiler type")
    compiled.__class__=classes[type(compiled)]
    compiled._conversation_enabled=enabled;compiled._conversation_domain=domain;compiled._conversation_workflow=workflow
    compiled._conversation_workflows=workflows
    return compiled
