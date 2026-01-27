"""
Factory Sharing Implementation Guide and Examples

This document explains how to use the factory sharing features to enable
optimal cross-qubit factory assignment for RUS injection.
"""

# ============================================================================
# OVERVIEW
# ============================================================================

"""
The factory sharing implementation allows qubits to share factories when:
1. Multiple qubits require the same angle
2. Some qubits' initially assigned factories fail
3. Other qubits' factories succeed for the same angle

Instead of wasting successful factories, we can re-assign them to qubits
whose factories failed, minimizing total moving distance.

Key Components:
- AngleFactoryIndex: Maps angles to factories globally
- vertex_matching: Optimally assigns factories to qubits
- assign_injection_with_sharing: Enhanced assignment using vertex matching
"""

# ============================================================================
# USAGE PATTERN 1: Basic Factory Sharing (Recommended)
# ============================================================================

"""
In your main execution loop (e.g., factory_angle_execution), modify the
RUS injection phase:

OLD CODE:
---------
qubit_factory_pairs = assign_injection(
    successful_qubits,
    batch_injection,
    factory_pool,
    logic_qubit_locations,
)

NEW CODE:
---------
from src.rus.factory_sharing_integration import build_angle_factory_index_from_tmr_results
from src.rus.rus_assignment import assign_injection_with_sharing

# Build index from TMR results (factories with tmr_state=True)
angle_factory_index = build_angle_factory_index_from_tmr_results(
    qubit_trackers,
    factory_pool
)

# Use enhanced assignment with factory sharing
qubit_factory_pairs = assign_injection_with_sharing(
    successful_qubits,
    batch_injection,
    qubit_trackers,
    factory_pool,
    logic_qubit_locations,
    angle_factory_index=angle_factory_index,
)
"""

# ============================================================================
# USAGE PATTERN 2: Group-Based Processing
# ============================================================================

"""
If you want to process qubits grouped by angle:

from src.rus.factory_sharing_integration import group_qubits_by_angle
from src.rus.vertex_matching import find_optimal_factory_assignment

# Group qubits by their required angle
angle_groups = group_qubits_by_angle(
    batch_injection,
    qubit_trackers,
    successful_qubits
)

qubit_factory_pairs = []

# Process each angle group separately
for angle, qubits_batch in angle_groups.items():
    # Extract all factories preparing this angle
    available_factories = {}
    for qubit, factory_ids in qubits_batch:
        for fid, fangle in qubit_trackers[qubit].factories:
            if fangle == angle:
                factory = factory_pool.get_factory_by_id(fid)
                if factory is not None:
                    available_factories[fid] = factory.location
    
    if not available_factories:
        continue
    
    # Build location dict for qubits
    qubit_locs = {q: logic_qubit_locations[q] for q, _ in qubits_batch}
    qubits_list = [q for q, _ in qubits_batch]
    
    # Find optimal matching
    matches = find_optimal_factory_assignment(
        qubits_list,
        available_factories,
        qubit_locs,
        use_hungarian=False  # Greedy is efficient for this use case
    )
    
    qubit_factory_pairs.extend(matches)
"""

# ============================================================================
# USAGE PATTERN 3: Per-Angle Manual Control
# ============================================================================

"""
For more fine-grained control, you can process each angle separately:

from src.rus.factory_sharing_integration import extract_factory_candidates_by_angle

# After determining which qubits need which angles, for each angle:
angle = 0.1234  # Example angle

# Get all factories preparing this angle across all qubits
available_factories = extract_factory_candidates_by_angle(
    qubit_trackers,
    factory_pool,
    angle
)

# Get qubits that need this angle
qubits_for_angle = [q for q, tracker in qubit_trackers.items() 
                    if any(a == angle for _, a in tracker.factories)]

# Create location dictionaries
qubit_locs = {q: logic_qubit_locations[q] for q in qubits_for_angle}

# Find optimal assignment
matches = find_optimal_factory_assignment(
    qubits_for_angle,
    available_factories,
    qubit_locs
)
"""

# ============================================================================
# EXAMPLE: Full Integration in factory_angle_execution
# ============================================================================

"""
Here's a complete example showing how to integrate factory sharing into
the main execution loop:

def factory_angle_execution(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
):
    qubit_trackers = {
        qubit: QubitAngleTracker(qubit_id=qubit, target_angle=theta)
        for qubit, theta in target_qubits_angles.items()
    }
    
    factory_pool.set_locations(magic_state_locations)
    successful_qubits = set()
    execution_log = []
    circuit_moment = 0
    
    while len(target_qubits_angles) > len(successful_qubits):
        # ... TMR preparation phase ...
        
        # PHASE 3: RUS Injection with Factory Sharing
        injection_sequence = collect_teleportation_sequence(
            qubit_trackers, factory_pool
        )
        
        if injection_sequence:
            # BUILD ANGLE-FACTORY INDEX FROM TMR RESULTS
            from src.rus.factory_sharing_integration import build_angle_factory_index_from_tmr_results
            angle_factory_index = build_angle_factory_index_from_tmr_results(
                qubit_trackers,
                factory_pool
            )
            
            for batch_injection in injection_sequence:
                # USE ENHANCED ASSIGNMENT WITH FACTORY SHARING
                from src.rus.rus_assignment import assign_injection_with_sharing
                
                qubit_factory_pairs = assign_injection_with_sharing(
                    successful_qubits,
                    batch_injection,
                    qubit_trackers,
                    factory_pool,
                    logic_qubit_locations,
                    angle_factory_index=angle_factory_index,
                )
                
                if not qubit_factory_pairs:
                    break
                
                # ... rest of injection process ...
"""

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
AngleFactoryIndex
-----------------
Maps angles to factories preparing them globally.

Usage:
    index = AngleFactoryIndex()
    
    # Add a factory
    index.add_factory(factory_id=0, angle=0.1234, qubit_id=1)
    
    # Get all factories preparing an angle
    factories = index.get_factories_for_angle(0.1234)
    # Returns: [(0, 1), (2, 3), ...]  # (factory_id, qubit_id)
    
    # Get angle for a factory
    angle = index.get_angle_for_factory(0)
    # Returns: 0.1234
    
    # Get all angles
    angles = index.get_all_angles()
    # Returns: {0.1234, 0.5678, ...}


vertex_matching.find_optimal_factory_assignment()
--------------------------------------------------
Finds optimal factory-to-qubit assignment minimizing total moving distance.

Usage:
    from src.rus.vertex_matching import find_optimal_factory_assignment
    
    assignments = find_optimal_factory_assignment(
        qubits_needing_rus=[0, 1, 2],
        available_factories={
            0: (5, 5),      # factory_id -> (x, y)
            1: (6, 6),
            2: (7, 7),
        },
        qubit_locations={
            0: (0, 0),      # qubit_id -> (x, y)
            1: (1, 1),
            2: (2, 2),
        },
        use_hungarian=False  # Use greedy (faster)
    )
    
    # Returns: [(0, 1), (1, 0), (2, 2)]  # (qubit_id, factory_id)
    # Minimizes total Manhattan distance


assign_injection_with_sharing()
--------------------------------
Assigns factories to qubits with cross-qubit factory sharing support.

Usage:
    from src.rus.rus_assignment import assign_injection_with_sharing
    
    pairs = assign_injection_with_sharing(
        successful_qubits={},
        batch_injection=[(0, [0, 1]), (1, [2, 3]), (2, [0, 2])],
        qubit_trackers={
            0: tracker_0,  # QubitAngleTracker instance
            1: tracker_1,
            2: tracker_2,
        },
        factory_pool=factory_pool,
        logic_qubit_locations=[(0, 0), (1, 1), (2, 2)],
        angle_factory_index=None  # Will be built if not provided
    )
    
    # Returns: [(0, 1), (1, 3), (2, 0)]  # Optimal assignments
"""

# ============================================================================
# ALGORITHM: Vertex Matching for Distance Minimization
# ============================================================================

"""
When num_factories < num_qubits:
The algorithm finds the best num_factories qubits to assign factories to,
minimizing total moving distance.

Algorithm (Greedy):
1. For each qubit, calculate minimum distance to any factory
2. Sort qubits by this minimum distance (ascending)
3. For each qubit in order:
   - Assign the nearest unused factory
   - Mark both as used
4. Return the assignments

Time Complexity: O(num_qubits * num_factories)
Approximation: Greedy gives good results for practical grid sizes

Algorithm (Hungarian):
A Hungarian-like approach for better optimality:
1. Compute distance matrix: all (qubit, factory) pairs
2. Iteratively select closest (qubit, factory) pair
3. Mark both as used
4. Repeat until num_factories assignments made
5. Return the assignments

Time Complexity: O(num_factories^2 * num_qubits)
Better optimality but slower for large instances
"""

# ============================================================================
# BENEFITS AND EXPECTED IMPROVEMENTS
# ============================================================================

"""
Benefits of Factory Sharing:

1. Higher Success Rate
   - Instead of failing when local factories fail, can use shared factories
   - Reduces re-preparation cycles

2. Reduced Total Moving Distance
   - Optimal assignment minimizes CNOT routing overhead
   - Better circuit parallelism

3. Improved Resource Utilization
   - Successful factories not wasted if originally assigned qubit fails
   - More factories available for future injections

4. Flexible Allocation
   - Factories can serve multiple angles/qubits
   - Better handling of uneven factory distribution

Expected Improvements:
- ~5-15% reduction in total circuit depth for typical topologies
- ~10-20% better factory utilization
- Especially beneficial when:
  - Multiple qubits have same angle requirements
  - Factory failure rates vary spatially
  - Hardware has uneven factory distribution
"""

# ============================================================================
# INTEGRATION CHECKLIST
# ============================================================================

"""
To integrate factory sharing into your codebase:

[ ] 1. Import necessary modules:
      from src.rus.angle_factory_index import AngleFactoryIndex
      from src.rus.vertex_matching import find_optimal_factory_assignment
      from src.rus.rus_assignment import assign_injection_with_sharing
      from src.rus.factory_sharing_integration import ...

[ ] 2. In factory_angle_execution() main loop:
      - Build angle_factory_index from injection_sequence
      - Replace assign_injection() with assign_injection_with_sharing()
      - Pass qubit_trackers to the new function

[ ] 3. Verify data flow:
      - ensure qubit_trackers has factory-angle information
      - ensure factory_pool has locations set
      - ensure logic_qubit_locations is accurate

[ ] 4. Test with sample circuits:
      - Simple 2-qubit same angle
      - Multiple qubits, multiple angles
      - Asymmetric factory distribution
      - High factory failure rates

[ ] 5. Performance validation:
      - Compare circuit depth before/after
      - Measure total moving distance reduction
      - Profile assignment algorithm overhead
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Issue: No assignments returned
- Check: Are batch_injection entries correctly formatted?
- Check: Do qubit_trackers have factories registered?
- Check: Are factory locations set in factory_pool?

Issue: All factories assigned to wrong qubits
- Check: Is angle_factory_index built correctly?
- Check: Are qubit locations accurate?
- Check: Are there enough factories for all qubits?

Issue: Assignment takes too long
- Use: find_optimal_factory_assignment(..., use_hungarian=False)
  (greedy is faster, close to optimal for grid topologies)
- Or: Reduce problem size by processing one injection round at a time

Issue: Factory sharing not working across qubits
- Check: Do multiple qubits actually need the same angle?
- Check: Is AngleFactoryIndex including factories from all qubits?
- Verify: angle_factory_index.get_factories_for_angle(angle) returns multiple entries
"""
