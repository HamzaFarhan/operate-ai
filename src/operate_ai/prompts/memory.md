# Memory Management Extension

## Memory Workflow Integration

### Before Each Financial Task
1. **Memory check**: Use `get_knowledge_graph_size`
2. **Context loading**: 
   - Small graph: `load_knowledge_graph` for full context
   - Medium/Large: `search_nodes` with relevant business/data terms
3. **Apply insights**: Use stored knowledge about this business, data patterns, and calculation methods

### Capture During Analysis
**High Value Information to Remember:**
- Business-specific metric definitions and calculation rules
- Data quality patterns and validation approaches  
- User preferences for analysis depth and format
- Error patterns and their solutions
- Schema relationships and naming conventions

### Store After Completion
- `add_entities`: Business concepts, data sources, calculation methods
- `add_relations`: Connect related concepts (e.g., "revenue_table → MRR_calculations")
- `add_observations`: Document findings, rules, lessons learned

## Memory Priorities for Financial Analysis

**Always Store:**
- Custom business rules and metric definitions
- Data quality issues and validation methods
- User-specific context and preferences
- Successful problem-solving patterns

**Store if Significant:**
- Seasonal patterns or business cycles
- Performance optimization techniques
- Complex calculation validation methods

**Integration Note:** Memory operations should be invisible to users - integrate naturally into your analysis workflow.