import rusty_evolution

# Create a configuration object with your parameters.
config = rusty_evolution.PyEvolutionConfig(
    time_horizon_in_days=365,
    generations=100,
    population_size=50,
    simulations_per_generation=10,
    assets_under_management=4,
    money_to_invest=10000.0,
    risk_free_rate=0.02,
    elitism_rate=0.1,
    mutation_rate=0.05,
    tournament_size=3,
    sampler=your_sampler_object,  # Ensure this matches your expected Sampler configuration.
    generation_check_interval=10
)

# Run the evolution function.
result = rusty_evolution.run_evolution(config)

# Use the returned result.
print(result)
