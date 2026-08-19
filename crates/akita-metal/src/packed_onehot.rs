use akita_field::AkitaError;
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{CommitInnerWitness, PackedOneHotView};

use crate::backend::MetalCommitBackend;
use crate::field::F;
use crate::prepared::MetalPreparedSetup;
use crate::{MetalCommitError, MetalExecutionPolicy};

impl<const D: usize> RootCommitKernel<PackedOneHotView<'_, F, D>, F, D> for MetalCommitBackend {
    #[tracing::instrument(skip_all, name = "MetalCommitBackend::packed_onehot_commit_inner")]
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<PackedOneHotView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        let [source] = sources.as_slice() else {
            return self.unsupported_packed_or_cpu(
                prepared,
                sources,
                plan,
                "packed one-hot Metal commitment requires one physical source".into(),
            );
        };
        let Some(runtime) = self.runtime() else {
            return self.unsupported_packed_or_cpu(
                prepared,
                sources,
                plan,
                "no Metal runtime is available".into(),
            );
        };
        let shape = match crate::packed_onehot_fp128_d512::validate_shape(*source, plan) {
            Ok(shape) => shape,
            Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return self
                    .cpu_backend()
                    .commit_inner_group(&prepared.cpu, sources, plan);
            }
            Err(error) => return Err(error),
        };
        if !runtime.supports_packed_fp128_d512_panels() {
            return self.unsupported_packed_or_cpu(
                prepared,
                sources,
                plan,
                "fp128 D512 panel pipeline is unavailable".into(),
            );
        }
        crate::packed_onehot_fp128_d512::commit_validated(
            self, prepared, runtime, *source, plan, shape,
        )
        .map(|witness| vec![witness])
    }
}

impl MetalCommitBackend {
    fn unsupported_packed_or_cpu<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        sources: Vec<PackedOneHotView<'_, F, D>>,
        plan: CommitInnerPlan,
        reason: String,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        match self.policy() {
            MetalExecutionPolicy::RequireMetal => {
                Err(MetalCommitError::UnsupportedShape(reason).into_akita())
            }
            MetalExecutionPolicy::PreferMetal => {
                self.cpu_backend()
                    .commit_inner_group(&prepared.cpu, sources, plan)
            }
        }
    }
}
