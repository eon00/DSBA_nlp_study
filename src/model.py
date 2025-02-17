from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig): # config.model_config.model_name
        super().__init__()
        cofig = model_config.model_config
        # bert-base-uncased, ModernBERT-base
        self.model = AutoModel.from_pretrained(cofig.model_name, add_pooling_layer=False)

        # Dropout 추가 (config에서 dropout_rate 설정 가능하도록)
        self.dropout = nn.Dropout(cofig.dropout_rate)  # 기본값 0.1 정도 추천

        # 분류 헤드
        self.classification_head = nn.Linear(self.model.config.hidden_size, cofig.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    
    def forward(self, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor, 
    token_type_ids: Optional[torch.Tensor] = None,  # ✅ 선택적 인자로 변경
    label: Optional[torch.Tensor] = None  # ✅ 기본값 추가하여 선택적으로 사용 가능
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
        """
    # token_type_ids가 필요 없는 모델도 있음 → None 허용
        representation = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )

        # [CLS] 토큰을 사용하여 문장 표현 생성 (평균 풀링 대신)
        last_hidden_state = representation['last_hidden_state']
        pooled_output = torch.mean(last_hidden_state, dim=1)
        logits = self.classification_head(pooled_output)
        loss = self.loss_fn(logits, label) if label is not None else None
        return loss, logits