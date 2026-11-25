import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:crisisq_device_study/main.dart';
import 'package:crisisq_device_study/services/study_controller.dart';

void main() {
  testWidgets('Setup UI labels debug mode and prevents timing before parity', (
    tester,
  ) async {
    final c = StudyController();
    await tester.pumpWidget(CrisisQApp(controller: c, initialize: false));
    expect(find.text('Measure the frozen pipeline'), findsOneWidget);
    expect(find.text('Setup / debug build'), findsOneWidget);
    final button = find.widgetWithText(FilledButton, 'Start FP32 study');
    await tester.scrollUntilVisible(
      button,
      300,
      scrollable: find.byType(Scrollable).first,
    );
    expect(tester.widget<FilledButton>(button).onPressed, isNull);
  });
}
