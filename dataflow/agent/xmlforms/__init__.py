# XML表单驱动Agent模块
from .form_templates import FormTemplateManager
from .models import FormRequest, FormResponse, SimpleWorkflowXML

__all__ = [
    "FormTemplateManager",
    "FormRequest",
    "FormResponse", 
    "SimpleWorkflowXML"
]
