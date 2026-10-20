function Adapt.adapt_structure(to, op::WeightingOp)
  return WeightingOp(Adapt.adapt(to, op.weights), 1)
end

function Adapt.adapt_structure(to, op::NormalOpImpl)
  parent = Adapt.adapt(to, op.parent)
  weights = Adapt.adapt(to, op.weights)
  return NormalOpImpl(parent, weights)
end

function Adapt.adapt_structure(to, op::ProdOp)
  return ProdOp(Adapt.adapt(to, op.A), Adapt.adapt(to, op.B))
end

function Adapt.adapt_structure(to, op::DiagOp)
  adapted = DiagOp([Adapt.adapt(to, child) for child in op.ops]; scheduler = op.scheduler)
  adapted.equalOps = op.equalOps
  return adapted
end
