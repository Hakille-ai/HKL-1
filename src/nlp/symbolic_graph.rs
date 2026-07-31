//! Neuro-Symbolic Knowledge Graph & Hebbian Concept Binding module for HKL-1.
//! Stores symbolic concepts and triple relations (Subject, Relation, Object),
//! binding them to SNN neuron assemblies with spreading activation.

use crate::core::math::FixedPoint;

pub const MAX_CONCEPTS: usize = 32;
pub const MAX_TRIPLES: usize = 64;

/// Concept Node in Neuro-Symbolic Memory
#[derive(Clone, Copy)]
pub struct ConceptNode {
    pub id: u8,
    pub name: [u8; 16],
    pub name_len: u8,
    pub activation: FixedPoint,
    pub neuron_assembly_base: u16,
    pub valid: bool,
}

impl ConceptNode {
    pub const fn empty() -> Self {
        Self {
            id: 0,
            name: [0; 16],
            name_len: 0,
            activation: FixedPoint::ZERO,
            neuron_assembly_base: 0,
            valid: false,
        }
    }
}

/// Symbolic Triple Relation: (Subject, Relation, Object)
#[derive(Clone, Copy)]
pub struct SymbolicTriple {
    pub subject_id: u8,
    pub relation_id: u8, // e.g. 0="is_a", 1="has_a", 2="chases", 3="causes"
    pub object_id: u8,
    pub weight: FixedPoint,
    pub valid: bool,
}

impl SymbolicTriple {
    pub const fn empty() -> Self {
        Self {
            subject_id: 0,
            relation_id: 0,
            object_id: 0,
            weight: FixedPoint::ZERO,
            valid: false,
        }
    }
}

/// Spiking Neuro-Symbolic Knowledge Graph
pub struct SymbolicKnowledgeGraph {
    pub concepts: [ConceptNode; MAX_CONCEPTS],
    pub triples: [SymbolicTriple; MAX_TRIPLES],
    pub concept_count: usize,
    pub triple_count: usize,
}

impl SymbolicKnowledgeGraph {
    pub fn new() -> Self {
        Self {
            concepts: [ConceptNode::empty(); MAX_CONCEPTS],
            triples: [SymbolicTriple::empty(); MAX_TRIPLES],
            concept_count: 0,
            triple_count: 0,
        }
    }

    /// Add or retrieve concept node ID by name
    pub fn add_concept(&mut self, name: &[u8]) -> u8 {
        for i in 0..MAX_CONCEPTS {
            if self.concepts[i].valid {
                let len = self.concepts[i].name_len as usize;
                if len == name.len() && &self.concepts[i].name[..len] == name {
                    return i as u8;
                }
            }
        }

        // Add new concept slot
        for i in 0..MAX_CONCEPTS {
            if !self.concepts[i].valid {
                let mut name_buf = [0u8; 16];
                let len = name.len().min(16);
                name_buf[..len].copy_from_slice(&name[..len]);

                self.concepts[i] = ConceptNode {
                    id: i as u8,
                    name: name_buf,
                    name_len: len as u8,
                    activation: FixedPoint::ZERO,
                    neuron_assembly_base: (i * 16) as u16,
                    valid: true,
                };
                self.concept_count += 1;
                return i as u8;
            }
        }

        0
    }

    /// Add symbolic triple relation (Subject, Relation, Object)
    pub fn add_triple(&mut self, subj_name: &[u8], rel_id: u8, obj_name: &[u8]) {
        let subj_id = self.add_concept(subj_name);
        let obj_id = self.add_concept(obj_name);

        for i in 0..MAX_TRIPLES {
            if !self.triples[i].valid {
                self.triples[i] = SymbolicTriple {
                    subject_id: subj_id,
                    relation_id: rel_id,
                    object_id: obj_id,
                    weight: FixedPoint::ONE,
                    valid: true,
                };
                self.triple_count += 1;
                return;
            }
        }
    }

    /// Activate a concept and propagate spreading activation across relation edges
    pub fn activate_and_propagate(&mut self, concept_id: u8, initial_activation: FixedPoint) {
        if concept_id as usize >= MAX_CONCEPTS {
            return;
        }

        self.concepts[concept_id as usize].activation = initial_activation;

        // Spreading activation across triples
        for i in 0..MAX_TRIPLES {
            if self.triples[i].valid && self.triples[i].subject_id == concept_id {
                let target_obj = self.triples[i].object_id as usize;
                let spread =
                    initial_activation * self.triples[i].weight * FixedPoint::from_f32(0.7);
                self.concepts[target_obj].activation =
                    (self.concepts[target_obj].activation + spread).min(FixedPoint::ONE);
            }
        }
    }

    /// Decay concept activations over time
    pub fn decay_activations(&mut self, decay_rate: FixedPoint) {
        for c in &mut self.concepts {
            if c.valid {
                c.activation = c.activation * (FixedPoint::ONE - decay_rate);
            }
        }
    }
}
