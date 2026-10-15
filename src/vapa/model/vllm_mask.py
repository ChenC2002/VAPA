"""vLLM V1 bridge to the same token-prefix state machine used by Transformers.

Imported only by the optional vLLM worker. No structured-output grammar is added:
argument suffixes remain unconstrained, including malformed tool arguments.
"""

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from vapa.model.protocols import ActionPrefixMask
from vapa.model.transformers import _masked_logit_row


class ActionPrefixLogitsProcessor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    @classmethod
    def validate_params(cls, params):
        payload = (params.extra_args or {}).get("vapa_action_mask")
        if payload is None:
            raise ValueError("VAPA generation requires its action-prefix mask")
        ActionPrefixMask(**payload)

    def new_req_logits_processor(self, params):
        self.validate_params(params)
        mask = ActionPrefixMask(**params.extra_args["vapa_action_mask"])

        def apply(output_ids, logits):
            return _masked_logit_row(
                logits, mask.allowed_next(output_ids), mask, mask.forbidden_next(output_ids)
            )

        return apply
